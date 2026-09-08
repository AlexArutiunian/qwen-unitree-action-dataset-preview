# Google Apps Script backend for user action suggestions

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
