const SPREADSHEET_ID = '1gAKnIjSKDAmWRTAcVCFNIKOsUh9fqc7vngsEDbxhHCU';
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
