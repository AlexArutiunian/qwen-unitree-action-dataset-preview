from pathlib import Path

p = Path('web/index.html')
s = p.read_text()
old = '<aside>\n<span class="eyebrow">ACTION</span>'
new = '''<aside>
<section class="suggestion-box" aria-labelledby="suggestionHeading">
<span class="eyebrow">ПРЕДЛОЖИТЬ ACTION</span>
<h2 id="suggestionHeading">Что должен сделать робот?</h2>
<p class="suggestion-help">Опиши движение рук и пояса своими словами. Ноги в этой форме не двигаем.</p>
<form id="suggestionForm">
<textarea id="suggestionInput" minlength="8" maxlength="500" rows="4" required aria-describedby="suggestionHelp suggestionStatus" placeholder="Например: подними левую руку вверх, правую согни перед собой и немного поверни пояс вправо"></textarea>
<span id="suggestionHelp" class="suggestion-example">Можно писать по-русски или по-английски. Лучше описывать одно понятное действие.</span>
<div class="suggestion-footer"><span id="suggestionCount">0 / 500</span><button id="suggestionSubmit" type="submit">Отправить команду</button></div>
<input id="suggestionWebsite" class="suggestion-honeypot" name="website" type="text" tabindex="-1" autocomplete="off" aria-hidden="true">
<p id="suggestionStatus" class="suggestion-status" role="status"></p>
</form>
</section>
<div class="divider"></div><span class="eyebrow">ACTION</span>'''
if old not in s:
    raise SystemExit('index anchor not found')
p.write_text(s.replace(old, new, 1))

p = Path('web/src/main.js')
s = p.read_text()
import_anchor = "import { sample91 } from './sample.js';"
if import_anchor not in s:
    raise SystemExit('main import anchor not found')
s = s.replace(import_anchor, import_anchor + "\nimport { SUGGESTION_ENDPOINT } from './submission-config.js';", 1)

event_anchor = "$('play').onclick = toggle;"
block = r'''function syncSuggestionCount() {
  const input = $('suggestionInput');
  if (!input) return;
  $('suggestionCount').textContent = `${input.value.length} / 500`;
}
function suggestionEndpointReady() {
  return /^https:\/\/script\.google\.com\/macros\/s\/.+\/exec(?:\?|$)/.test(SUGGESTION_ENDPOINT);
}
function initSuggestionForm() {
  const input = $('suggestionInput'), submit = $('suggestionSubmit'), status = $('suggestionStatus');
  if (!input || !submit || !status) return;
  input.addEventListener('input', syncSuggestionCount);
  syncSuggestionCount();
  if (!suggestionEndpointReady()) {
    submit.disabled = true;
    status.textContent = 'Сбор команд готов, осталось подключить Google Apps Script endpoint.';
    status.classList.add('is-warning');
  } else {
    status.textContent = 'Команда сохранится в таблицу автора проекта.';
  }
}
async function submitSuggestion(event) {
  event.preventDefault();
  const input = $('suggestionInput'), submit = $('suggestionSubmit'), status = $('suggestionStatus');
  const command = input.value.replace(/\s+/g, ' ').trim();
  status.classList.remove('is-error', 'is-success', 'is-warning');
  if (!suggestionEndpointReady()) {
    status.textContent = 'Хранилище ещё не подключено.';
    status.classList.add('is-warning');
    return;
  }
  if (command.length < 8) {
    status.textContent = 'Опиши действие чуть подробнее.';
    status.classList.add('is-error');
    input.focus();
    return;
  }
  if (command.length > 500) {
    status.textContent = 'Команда длиннее 500 символов.';
    status.classList.add('is-error');
    return;
  }
  const submissionId = crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const payload = new URLSearchParams({
    command,
    current_action_id: String(currentAction?.sample_id ?? ''),
    current_action_class: String(currentAction?.class ?? ''),
    current_action_augmentation: String(currentAction?.augmentation_type ?? ''),
    page_url: location.href,
    submission_id: submissionId,
    website: $('suggestionWebsite')?.value || ''
  });
  submit.disabled = true;
  submit.textContent = 'Отправляем…';
  status.textContent = 'Сохраняем команду…';
  try {
    await fetch(SUGGESTION_ENDPOINT, { method: 'POST', mode: 'no-cors', body: payload });
    input.value = '';
    syncSuggestionCount();
    status.textContent = 'Спасибо! Команда сохранена для будущего датасета.';
    status.classList.add('is-success');
  } catch (error) {
    status.textContent = 'Не удалось отправить. Проверь соединение и попробуй ещё раз.';
    status.classList.add('is-error');
  } finally {
    submit.disabled = false;
    submit.textContent = 'Отправить команду';
  }
}
$('suggestionForm')?.addEventListener('submit', submitSuggestion);
initSuggestionForm();

'''
if event_anchor not in s:
    raise SystemExit('main event anchor not found')
p.write_text(s.replace(event_anchor, block + event_anchor, 1))

p = Path('web/src/style.css')
s = p.read_text()
append = r'''
.suggestion-box{padding:0}.suggestion-box h2{font-size:20px;line-height:1.25;margin:10px 0 7px}.suggestion-help{font-size:14px;line-height:1.45;color:#a6b5c6;margin:0 0 12px}.suggestion-box form{display:grid;gap:9px}.suggestion-box textarea{width:100%;min-height:112px;resize:vertical;background:#172334;color:#eef5fb;border:1px solid #3a4b61;border-radius:8px;padding:12px 13px;font:inherit;line-height:1.45;outline:none}.suggestion-box textarea::placeholder{color:#8293a7}.suggestion-box textarea:focus{border-color:#77dfce;box-shadow:0 0 0 2px #77dfce22}.suggestion-example{font-size:12px;line-height:1.4;color:#8297ad}.suggestion-footer{display:flex;align-items:center;justify-content:space-between;gap:10px}.suggestion-footer>span{font-size:12px;color:#8fa3b8;font-variant-numeric:tabular-nums;white-space:nowrap}.suggestion-footer button{background:#79d8c8;color:#10262a;border-color:#79d8c8;font-weight:700;min-height:42px}.suggestion-footer button:hover{background:#92e5d7}.suggestion-status{min-height:20px;margin:0;font-size:12px;line-height:1.4;color:#91a6ba}.suggestion-status.is-success{color:#82dfc5}.suggestion-status.is-error{color:#f0a5a5}.suggestion-status.is-warning{color:#d9bd83}.suggestion-honeypot{position:absolute!important;left:-10000px!important;width:1px!important;height:1px!important;opacity:0!important;pointer-events:none!important}@media(max-width:720px){.suggestion-box h2{font-size:20px}.suggestion-box textarea{min-height:128px;font-size:16px}.suggestion-footer{align-items:stretch;flex-direction:column}.suggestion-footer button{width:100%;min-height:48px;font-size:16px}}
'''
p.write_text(s + append)

Path('web/src/submission-config.js').write_text("// Public Google Apps Script Web App URL. No secrets belong here.\nexport const SUGGESTION_ENDPOINT = '';\n")

backend = Path('tools/google_apps_script')
backend.mkdir(parents=True, exist_ok=True)
backend.joinpath('Code.gs').write_text(r'''const SPREADSHEET_ID = '1gAKnIjSKDAmWRTAcVCFNIKOsUh9fqc7vngsEDbxhHCU';
const SHEET_NAME = 'Команды';

function doGet() {
  return ContentService.createTextOutput('G1 suggestion collector is running');
}

function doPost(e) {
  const p = (e && e.parameter) || {};
  if (String(p.website || '').trim()) return response_({ ok: true });

  const command = clean_(p.command, 500);
  if (command.length < 8) return response_({ ok: false, error: 'command_too_short' });

  const row = [
    new Date().toISOString(),
    safeCell_(command),
    safeCell_(clean_(p.current_action_id, 32)),
    safeCell_(clean_(p.current_action_class, 80)),
    safeCell_(clean_(p.current_action_augmentation, 80)),
    safeCell_(clean_(p.page_url, 500)),
    safeCell_(clean_(p.submission_id, 100)),
    'new',
    ''
  ];

  const lock = LockService.getScriptLock();
  if (!lock.tryLock(5000)) return response_({ ok: false, error: 'busy' });
  try {
    const sheet = SpreadsheetApp.openById(SPREADSHEET_ID).getSheetByName(SHEET_NAME);
    if (!sheet) throw new Error('Sheet not found: ' + SHEET_NAME);
    sheet.appendRow(row);
  } finally {
    lock.releaseLock();
  }
  return response_({ ok: true });
}

function clean_(value, maxLen) {
  return String(value || '').replace(/\s+/g, ' ').trim().slice(0, maxLen);
}

function safeCell_(value) {
  const text = String(value || '');
  return /^[=+\-@]/.test(text) ? "'" + text : text;
}

function response_(payload) {
  return ContentService
    .createTextOutput(JSON.stringify(payload))
    .setMimeType(ContentService.MimeType.JSON);
}
''')
backend.joinpath('README.md').write_text(r'''# Google Apps Script backend for user action suggestions

Target spreadsheet: `G1 пользовательские команды для дообучения`

Spreadsheet ID is already embedded in `Code.gs`.

Deploy once:

1. Create a Google Apps Script project.
2. Replace its `Code.gs` with this repository's `tools/google_apps_script/Code.gs`.
3. Deploy -> New deployment -> Web app.
4. Execute as: Me.
5. Who has access: Anyone.
6. Copy the `/exec` URL into `web/src/submission-config.js` as `SUGGESTION_ENDPOINT`.
7. Rebuild `demo/` with `npm run build` from `web/`.

The browser sends only the proposed command and current action context. The endpoint contains no private key or secret. The script rejects very short commands, limits lengths, uses a honeypot, serializes writes with LockService, and prevents spreadsheet formula injection.
''')
