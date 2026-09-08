from pathlib import Path

js_path = Path('web/src/main.js')
html_path = Path('web/index.html')
css_path = Path('web/src/style.css')

js = js_path.read_text()
html = html_path.read_text()
css = css_path.read_text()

marker = 'function sourceBrowseKey(action) {'
if marker in js:
    raise SystemExit('source-id browsing already installed')

needle = "function matchesAction(action, query) {\n  if (!query) return true;\n  const haystack = `${action.sample_id} ${action.source_id} ${action.split} ${action.class} ${action.augmentation_type} ${action.text}`.toLocaleLowerCase('ru');\n  return haystack.includes(query);\n}\n"
if needle not in js:
    raise SystemExit('matchesAction marker not found')

helpers = needle + r'''function sourceBrowseKey(action) {
  return action?.source_id ?? action?.sample_id;
}
function representativeScore(action) {
  const augmentation = String(action?.augmentation_type || '').toLowerCase();
  if (!augmentation || augmentation === 'original') return 0;
  if (Number(action?.sample_id) === Number(action?.source_id)) return 1;
  return 2;
}
function sourceRepresentatives() {
  const bySource = new Map();
  for (const action of actions) {
    const key = String(sourceBrowseKey(action));
    const current = bySource.get(key);
    if (!current || representativeScore(action) < representativeScore(current) ||
        representativeScore(action) === representativeScore(current) && action.sample_id < current.sample_id) {
      bySource.set(key, action);
    }
  }
  return [...bySource.values()].sort((a, b) => {
    const aKey = sourceBrowseKey(a), bKey = sourceBrowseKey(b);
    const aNumber = Number(aKey), bNumber = Number(bKey);
    if (Number.isFinite(aNumber) && Number.isFinite(bNumber) && aNumber !== bNumber) return aNumber - bNumber;
    return String(aKey).localeCompare(String(bKey), undefined, { numeric: true });
  });
}
function browseSource(offset) {
  const representatives = sourceRepresentatives();
  if (!representatives.length) return;
  const currentKey = String(sourceBrowseKey(currentAction));
  const index = representatives.findIndex(action => String(sourceBrowseKey(action)) === currentKey);
  const nextIndex = index + offset;
  if (nextIndex < 0 || nextIndex >= representatives.length) return;
  selectAction(representatives[nextIndex]);
  populateActions();
}
'''
js = js.replace(needle, helpers, 1)

old_status = "    $('datasetStatus').textContent = `Загружено ${actions.length} actions · train + validation + reserved`;"
new_status = "    $('datasetStatus').textContent = `Загружено ${sourceRepresentatives().length} source_id · ${actions.length} actions`;"
if old_status not in js:
    raise SystemExit('dataset status marker not found')
js = js.replace(old_status, new_status, 1)

old_nav = r'''$('prevAction').onclick = () => {
  const index = actions.findIndex(action => action.sample_id === currentAction.sample_id); if (index > 0) { selectAction(actions[index - 1]); populateActions(); }
};
$('nextAction').onclick = () => {
  const index = actions.findIndex(action => action.sample_id === currentAction.sample_id); if (index >= 0 && index < actions.length - 1) { selectAction(actions[index + 1]); populateActions(); }
};'''
new_nav = r'''$('prevAction').onclick = () => browseSource(-1);
$('nextAction').onclick = () => browseSource(1);'''
if old_nav not in js:
    raise SystemExit('navigation marker not found')
js = js.replace(old_nav, new_nav, 1)

js = js.replace("    status.textContent = 'Команда сохранится в таблицу автора проекта.';", "    status.textContent = '';", 1)
js = js.replace("    submit.textContent = 'Отправить команду';", "    submit.textContent = 'Отправить';", 1)
js_path.write_text(js)

html = html.replace('rows="4"', 'rows="2"', 1)
html = html.replace(
    'placeholder="Например: подними левую руку вверх, правую согни перед собой и немного поверни пояс вправо"',
    'placeholder="Например: подними левую руку, согни правую и поверни пояс вправо"',
    1,
)
html = html.replace('>Отправить команду</button>', '>Отправить</button>', 1)
html = html.replace('title="Предыдущий action" aria-label="Предыдущий action"', 'title="Предыдущий source_id" aria-label="Предыдущий source_id"')
html = html.replace('title="Следующий action" aria-label="Следующий action"', 'title="Следующий source_id" aria-label="Следующий source_id"')
html = html.replace('title="Предыдущий action">← Пред.</button>', 'title="Предыдущий source_id">← Пред.</button>')
html = html.replace('title="Следующий action">След. →</button>', 'title="Следующий source_id">След. →</button>')
html_path.write_text(html)

css_marker = '/* Source-id browsing + compact prompt polish v3 */'
if css_marker in css:
    raise SystemExit('UI polish already installed')

css += r'''

/* Source-id browsing + compact prompt polish v3 */
.suggestion-hero{
  min-height:96px;
  padding:12px 18px;
  gap:12px 18px;
  background:linear-gradient(180deg,#111d2a 0%,#101a25 100%);
  border-bottom:1px solid #3a4c61;
  box-shadow:0 7px 22px #00000024
}
.suggestion-hero::before{width:3px;background:linear-gradient(180deg,#8ce0d1,#5fb8c9)}
.suggestion-hero h2{font-size:18px;font-weight:720;color:#f3f7fb;letter-spacing:-.025em}
.suggestion-hero form{gap:5px 9px}
.suggestion-hero textarea{
  height:58px;
  min-height:58px;
  max-height:92px;
  border-radius:12px;
  padding:16px 15px;
  background:linear-gradient(180deg,#1a2a3e,#172638);
  border:1px solid #4a6079;
  box-shadow:inset 0 1px 0 #ffffff0d,0 3px 12px #00000020;
  font-size:15px;
  line-height:1.35
}
.suggestion-hero textarea::placeholder{color:#8fa5bb}
.suggestion-hero textarea:focus{border-color:#8ce0d1;box-shadow:0 0 0 3px #77dfce20,inset 0 1px 0 #ffffff10}
.suggestion-hero .suggestion-footer button{
  height:58px;
  min-height:58px;
  min-width:118px;
  border-radius:12px;
  padding:0 18px;
  background:linear-gradient(180deg,#91e5d6,#72d2c2);
  border-color:#91e5d6;
  box-shadow:0 5px 16px #45b8a52b;
  font-weight:760
}
.suggestion-hero .suggestion-footer button:hover{background:linear-gradient(180deg,#a0ecdf,#82dccd)}
.suggestion-hero .suggestion-status{min-height:0;font-size:11px}
.suggestion-hero .suggestion-status:empty{display:none}

@media(max-width:720px){
  .suggestion-hero{
    min-height:0;
    padding:9px 10px 10px;
    gap:6px;
    background:linear-gradient(180deg,#111d2a,#101923)
  }
  .suggestion-hero h2{font-size:15px;line-height:1.15;font-weight:720}
  .suggestion-hero form{gap:4px 7px}
  .suggestion-hero textarea{
    height:56px;
    min-height:56px;
    max-height:56px;
    padding:10px 12px;
    border-radius:11px;
    font-size:14px;
    line-height:1.25
  }
  .suggestion-hero .suggestion-footer button{
    height:56px;
    min-height:56px;
    min-width:88px;
    padding:0 12px;
    border-radius:11px;
    font-size:13px
  }
  .suggestion-hero .suggestion-status{font-size:10px;line-height:1.15}
  .stage-title{height:108px;padding:10px 138px 8px 12px}
  .stage-title .eyebrow{font-size:9px}
  .stage-title h1{
    font-size:15px;
    line-height:1.12;
    margin:4px 0 4px;
    max-width:100%;
    -webkit-line-clamp:3
  }
  .stage-title p{font-size:10.5px;margin:3px 0 0}
}
@media(max-width:380px){
  .suggestion-hero{padding-inline:8px}
  .suggestion-hero h2{font-size:14px}
  .suggestion-hero textarea{font-size:13.5px;padding-inline:10px}
  .suggestion-hero .suggestion-footer button{min-width:80px;padding-inline:9px;font-size:12px}
  .stage-title{height:104px;padding-right:128px}
  .stage-title h1{font-size:14px;line-height:1.1}
}
'''
css_path.write_text(css)
