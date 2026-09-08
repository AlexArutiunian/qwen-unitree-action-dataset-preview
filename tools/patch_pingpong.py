from pathlib import Path

main = Path('web/src/main.js')
text = main.read_text()

replacements = [
    (
        "let worker, timer, playing = false, time = 0, duration = 1.5, ready = false, modelReady = false, busy = false, wanted = 0, motionVersion = 1;",
        "let worker, timer, playing = false, playbackDirection = 1, time = 0, duration = 1.5, ready = false, modelReady = false, busy = false, wanted = 0, motionVersion = 1;"
    ),
    (
        "function toggle() {\n  if (!ready || duration <= 0) return;\n  if (time >= duration) requestPose(0);\n  playing = !playing;\n  syncPlaybackUi();\n}\nfunction reset() {\n  playing = false;\n  syncPlaybackUi();\n  requestPose(0);\n}",
        "function toggle() {\n  if (!ready || duration <= 0) return;\n  playing = !playing;\n  syncPlaybackUi();\n}\nfunction reset() {\n  playing = false;\n  playbackDirection = 1;\n  syncPlaybackUi();\n  requestPose(0);\n}\nfunction pingPongTime(delta) {\n  if (!Number.isFinite(delta) || delta <= 0 || duration <= 0) return time;\n  const cycle = duration * 2;\n  let phase = playbackDirection > 0 ? time : cycle - time;\n  phase = (phase + delta) % cycle;\n  if (phase < duration) {\n    playbackDirection = 1;\n    return phase;\n  }\n  if (phase > duration) {\n    playbackDirection = -1;\n    return cycle - phase;\n  }\n  playbackDirection = -1;\n  return duration;\n}"
    ),
    (
        "currentAction = action; motionVersion += 1; ready = false; busy = false; wanted = 0; time = 0; duration = 0; initialValues = null; unsupported = new Set();",
        "currentAction = action; motionVersion += 1; ready = false; busy = false; wanted = 0; time = 0; duration = 0; playbackDirection = 1; initialValues = null; unsupported = new Set();"
    ),
    (
        "$('seek').oninput = event => { playing = false; syncPlaybackUi(); requestPose(Number(event.target.value)); };",
        "$('seek').oninput = event => {\n  playing = false;\n  const value = Number(event.target.value);\n  if (value <= 0) playbackDirection = 1;\n  else if (value >= duration) playbackDirection = -1;\n  syncPlaybackUi();\n  requestPose(value);\n};"
    ),
    (
        "worker?.terminate(); clearTimeout(timer); ready = false; modelReady = false; busy = false; initialValues = null; playing = false; time = 0; wanted = 0; syncPlaybackUi(); setControls(false); clock();",
        "worker?.terminate(); clearTimeout(timer); ready = false; modelReady = false; busy = false; initialValues = null; playing = false; playbackDirection = 1; time = 0; wanted = 0; syncPlaybackUi(); setControls(false); clock();"
    ),
    (
        "duration = data.duration; $('seek').max = Math.max(duration, .001); time = 0; wanted = 0; busy = false; initialValues = null; unsupported = new Set(data.unsupported || []);",
        "duration = data.duration; $('seek').max = Math.max(duration, .001); time = 0; wanted = 0; playbackDirection = 1; busy = false; initialValues = null; unsupported = new Set(data.unsupported || []);"
    ),
    (
        "$('stageMeta').textContent = `${currentAction.split} · ${currentAction.class} · ${currentAction.augmentation_type || 'original'} · ${duration.toFixed(2)} с · loop`;",
        "$('stageMeta').textContent = `${currentAction.split} · ${currentAction.class} · ${currentAction.augmentation_type || 'original'} · ${duration.toFixed(2)} с · ping-pong`;"
    ),
    (
        "if (playing && ready && duration > 0) {\n      const next = time + elapsed * Number($('speed').value);\n      requestPose(next >= duration ? next % duration : next);\n    }",
        "if (playing && ready && duration > 0) {\n      requestPose(pingPongTime(elapsed * Number($('speed').value)));\n    }"
    ),
]

for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'Expected exactly one match, got {count}: {old[:80]!r}')
    text = text.replace(old, new)

main.write_text(text)

index = Path('web/index.html')
html = index.read_text()
old = 'После загрузки action движение запускается автоматически и повторяется по кругу. Кнопка ▶ / Ⅱ сверху ставит воспроизведение на паузу и возобновляет его.'
new = 'После загрузки action движение запускается автоматически в режиме ping-pong: плавно идёт вперёд до конца, затем тем же движением назад до начала и снова вперёд. Кнопка ▶ / Ⅱ сверху ставит воспроизведение на паузу и возобновляет его без смены направления.'
if html.count(old) != 1:
    raise SystemExit('Expected playback description exactly once')
index.write_text(html.replace(old, new))
