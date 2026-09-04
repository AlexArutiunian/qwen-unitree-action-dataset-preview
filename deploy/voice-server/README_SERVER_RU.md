# G1 Voice: сервер + робот

Минимальная схема:

```text
X на геймпаде → микрофон G1 → WebSocket → сервер
сервер: Whisper → Ollama → Supertonic → PCM 16 kHz
PCM → WebSocket → AudioClient.PlayStream → динамик G1
```

## Поведение X

- модель молчит: X начинает запись;
- запись идёт: тишина завершает автоматически, повторный X завершает вручную;
- модель думает или говорит: X сразу посылает cancel, вызывает `PlayStop` и начинает новую запись.

## Сервер

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

В `config.json` выставить модель Ollama и параметры ASR. Серверный токен не хранится
в файле: перед запуском задайте его через окружение (имя переменной определяется
полем `token_env`, по умолчанию `ROBOT_VOICE_TOKEN`):

```bash
export ROBOT_VOICE_TOKEN='replace-with-a-new-random-token'
```

```bash
./run_tmux.sh
tmux capture-pane -pt g1_voice_server -S -100
```

Проверка:

```bash
curl http://127.0.0.1:8765/health
```

## Робот

Пакеты ОС:

```bash
sudo apt update
sudo apt install -y python3-venv portaudio19-dev tmux
```

Окружение создаётся с доступом к уже установленному Unitree SDK:

```bash
cd robot
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Посмотреть USB-микрофон и геймпад:

```bash
python robot_client.py --list-devices
```

В `config.json` задать:

- `server_url`: `ws://IP_СЕРВЕРА:8765/ws`;
- тот же токен в клиентской конфигурации (на стороне сервера он берётся из окружения);
- `mic_device`: номер или имя USB-микрофона, либо `null` для устройства по умолчанию;
- `button_device`: путь `/dev/input/event...` или часть имени геймпада;
- `button_code`: для X у Xbox-подобного геймпада обычно `308` (`BTN_WEST`).

Доступ к input device:

```bash
sudo usermod -aG input $USER
```

После этого перелогиниться либо временно запустить через sudo с сохранением окружения.

Запуск:

```bash
./run_tmux.sh
tmux capture-pane -pt g1_voice_robot -S -100
```

## Локальная и глобальная сеть

Код одинаковый. Меняется только `server_url`:

```text
LAN:       ws://192.168.x.x:8765/ws
Tailscale: ws://100.x.x.x:8765/ws
TLS:       wss://voice.example.com/ws
```

Для интернета предпочтительнее Tailscale/ZeroTier или reverse proxy с TLS. Не открывайте обычный `ws://` напрямую в интернет. Токен обязателен.

## Настройка тишины

Если запись обрывается слишком рано, увеличить:

```json
"silence_ms": 900
```

Если тишина не распознаётся, увеличить `silence_rms`; если голос не обнаруживается — уменьшить.
