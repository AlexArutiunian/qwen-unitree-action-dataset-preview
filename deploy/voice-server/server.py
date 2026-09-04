#!/usr/bin/env python3
"""ASR + Ollama VLM + TTS server for Unitree G1. Supports optional RGB frame per turn."""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import soxr
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from supertonic import TTS

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8-sig"))
TOKEN_ENV = str(CFG.get("token_env", "ROBOT_VOICE_TOKEN"))
ACCESS_TOKEN = os.environ.get(TOKEN_ENV, "").strip()

SYSTEM = (
    "Ты голосовой помощник робота. Отвечай по-русски кратко, ясно и естественно. "
    "Возвращай только текст, который нормально звучит вслух: без Markdown, таблиц, эмодзи, "
    "кода и служебных пометок. Математические формулы сразу произноси словами по смыслу, "
    "сохраняя порядок действий; обычное тире не называй минусом. "
    "Не используй обращения вроде братан, если пользователь сам этого не просит. "
    "Обычно отвечай двумя-пятью предложениями. /no_think"
)


def speech_clean(text: str) -> str:
    """Only remove presentation markup. Never replace symbols with spoken words."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = text.replace("**", "").replace("__", "").replace("~~", "")
    return re.sub(r"\s+", " ", text).strip()


class SentenceBuffer:
    def __init__(self, limit: int = 260):
        self.buf = ""
        self.limit = limit

    def feed(self, token: str) -> list[str]:
        self.buf += token
        out: list[str] = []
        while True:
            m = re.search(r"^(.+?[.!?])(?=\s|$)", self.buf, flags=re.S)
            if m:
                out.append(m.group(1).strip())
                self.buf = self.buf[m.end():].lstrip()
                continue
            if len(self.buf) >= self.limit:
                cut = max(self.buf.rfind(x, 0, self.limit) for x in ("; ", ", ", " "))
                cut = cut if cut > 80 else self.limit
                out.append(self.buf[:cut].strip())
                self.buf = self.buf[cut:].lstrip()
                continue
            break
        return [x for x in out if x]

    def flush(self) -> list[str]:
        tail, self.buf = self.buf.strip(), ""
        return [tail] if tail else []


class Engine:
    def __init__(self) -> None:
        print("[LOAD] faster-whisper:", CFG["asr_model"], flush=True)
        try:
            self.asr = WhisperModel(
                CFG["asr_model"], device=CFG["asr_device"], compute_type=CFG["asr_compute"]
            )
        except Exception as exc:
            print(f"[ASR] GPU init failed: {exc}; fallback CPU int8", flush=True)
            self.asr = WhisperModel(CFG["asr_model"], device="cpu", compute_type="int8")

        print("[LOAD] Supertonic", flush=True)
        self.tts = TTS(auto_download=True)
        self.style = self.tts.get_voice_style(voice_name=CFG["tts_voice"])

    def transcribe(self, pcm: bytes) -> str:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        segments, _ = self.asr.transcribe(
            audio,
            language=CFG["language"],
            beam_size=int(CFG.get("asr_beam_size", 3)),
            vad_filter=bool(CFG.get("asr_vad_filter", False)),
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def synth_pcm16(self, text: str) -> bytes:
        wav, _ = self.tts.synthesize(
            text=speech_clean(text),
            voice_style=self.style,
            total_steps=int(CFG["tts_steps"]),
            speed=float(CFG["tts_speed"]),
            max_chunk_length=300,
            silence_duration=0.05,
            lang=CFG["language"],
            verbose=False,
        )
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        audio = soxr.resample(audio, 44100, 16000)
        return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


engine: Engine | None = None
app = FastAPI()


@app.on_event("startup")
async def startup() -> None:
    global engine
    if not ACCESS_TOKEN:
        raise RuntimeError(f"Required access token is missing: set {TOKEN_ENV}")
    engine = await asyncio.to_thread(Engine)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": engine is not None, "model": CFG["model"]}


@app.websocket("/ws")
async def robot_socket(ws: WebSocket) -> None:
    if ws.headers.get("x-token") != ACCESS_TOKEN:
        await ws.close(code=4401)
        return
    await ws.accept()
    print("[NET] robot connected", flush=True)

    history: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
    audio = bytearray()
    recording = False
    pending_image_b64: str | None = None
    turn_task: asyncio.Task | None = None
    send_lock = asyncio.Lock()

    async def send_json(data: dict[str, Any]) -> None:
        async with send_lock:
            await ws.send_text(json.dumps(data, ensure_ascii=False))

    async def send_pcm(data: bytes) -> None:
        async with send_lock:
            await ws.send_bytes(data)

    async def cancel_turn() -> None:
        nonlocal turn_task
        if turn_task and not turn_task.done():
            turn_task.cancel()
            await asyncio.gather(turn_task, return_exceptions=True)
        turn_task = None
        await send_json({"type": "audio_stop"})

    async def run_turn(pcm: bytes, image_b64: str | None = None) -> None:
        assert engine is not None
        stream_id = uuid.uuid4().hex
        try:
            await send_json({"type": "state", "value": "transcribing"})
            user_text = await asyncio.to_thread(engine.transcribe, pcm)
            if not user_text:
                await send_json({"type": "error", "text": "Речь не распознана"})
                await send_json({"type": "state", "value": "idle"})
                return
            await send_json({"type": "user", "text": user_text})
            await send_json({"type": "state", "value": "thinking"})

            user_message: dict[str, Any] = {"role": "user", "content": user_text}
            if image_b64:
                user_message["images"] = [image_b64]
                await send_json({"type": "info", "text": "RGB-кадр передан модели"})

            speech_q: asyncio.Queue[str | None] = asyncio.Queue()
            answer_parts: list[str] = []
            chunker = SentenceBuffer()

            async def llm() -> None:
                payload = {
                    "model": CFG["model"],
                    "messages": history + [user_message],
                    "stream": True,
                    "think": False,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": CFG["temperature"],
                        "num_ctx": CFG["num_ctx"],
                        "num_predict": CFG["num_predict"],
                    },
                }
                timeout = aiohttp.ClientTimeout(total=900, sock_read=900)
                async with aiohttp.ClientSession(timeout=timeout) as client:
                    async with client.post(CFG["ollama_url"].rstrip("/") + "/api/chat", json=payload) as resp:
                        resp.raise_for_status()
                        async for raw in resp.content:
                            for line in raw.splitlines():
                                if not line:
                                    continue
                                item = json.loads(line)
                                token = item.get("message", {}).get("content", "")
                                if token:
                                    answer_parts.append(token)
                                    for sentence in chunker.feed(token):
                                        await speech_q.put(sentence)
                                if item.get("done"):
                                    break
                for sentence in chunker.flush():
                    await speech_q.put(sentence)
                await speech_q.put(None)

            async def speaker() -> None:
                started = False
                while True:
                    sentence = await speech_q.get()
                    if sentence is None:
                        break
                    pcm16 = await asyncio.to_thread(engine.synth_pcm16, sentence)
                    if not pcm16:
                        continue
                    if not started:
                        await send_json({"type": "audio_start", "stream_id": stream_id})
                        await send_json({"type": "state", "value": "speaking"})
                        started = True
                    chunk_bytes = 16000 * 2 * int(CFG["audio_chunk_ms"]) // 1000
                    for pos in range(0, len(pcm16), chunk_bytes):
                        await send_pcm(pcm16[pos:pos + chunk_bytes])
                if started:
                    await send_json({"type": "audio_end", "stream_id": stream_id})

            tasks = [asyncio.create_task(llm()), asyncio.create_task(speaker())]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            answer = "".join(answer_parts).strip()
            if answer:
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": answer})
                await send_json({"type": "assistant", "text": answer})
            max_messages = int(CFG["max_history_messages"])
            if len(history) > max_messages + 1:
                history[:] = [history[0]] + history[-max_messages:]
            await send_json({"type": "state", "value": "idle"})
        except asyncio.CancelledError:
            await send_json({"type": "audio_stop"})
            raise
        except Exception as exc:
            print("[TURN]", repr(exc), flush=True)
            await send_json({"type": "error", "text": str(exc)})
            await send_json({"type": "audio_stop"})
            await send_json({"type": "state", "value": "idle"})

    try:
        await send_json({"type": "state", "value": "idle"})
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if recording:
                    audio.extend(msg["bytes"])
                continue
            if msg.get("text") is None:
                continue
            cmd = json.loads(msg["text"])
            kind = cmd.get("type")
            if kind == "start":
                await cancel_turn()
                audio.clear()
                image = cmd.get("image")
                pending_image_b64 = image if isinstance(image, str) and image else None
                recording = True
                await send_json({"type": "state", "value": "recording"})
            elif kind == "end" and recording:
                recording = False
                image_for_turn = pending_image_b64
                pending_image_b64 = None
                if len(audio) >= 3200:
                    turn_task = asyncio.create_task(run_turn(bytes(audio), image_for_turn))
                else:
                    await send_json({"type": "error", "text": "Слишком короткая запись"})
                    await send_json({"type": "state", "value": "idle"})
            elif kind == "cancel":
                recording = False
                pending_image_b64 = None
                audio.clear()
                await cancel_turn()
                await send_json({"type": "state", "value": "idle"})
            elif kind == "reset":
                history[:] = [history[0]]
                await send_json({"type": "info", "text": "История очищена"})
    except WebSocketDisconnect:
        pass
    finally:
        if turn_task and not turn_task.done():
            turn_task.cancel()
        print("[NET] robot disconnected", flush=True)


if __name__ == "__main__":
    uvicorn.run(app, host=CFG["bind"], port=int(CFG["port"]), log_level="info")
