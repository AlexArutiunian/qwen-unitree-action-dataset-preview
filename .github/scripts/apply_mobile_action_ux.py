from pathlib import Path
import re

html_path = Path('web/index.html')
js_path = Path('web/src/main.js')
css_path = Path('web/src/style.css')

html = html_path.read_text()
suggestion = re.search(
    r'(<section class="suggestion-box" aria-labelledby="suggestionHeading">.*?</section>)\n<div class="divider"></div>',
    html,
    re.S,
)
if not suggestion:
    raise SystemExit('suggestion block not found')
suggestion_block = suggestion.group(1).replace(
    'class="suggestion-box"',
    'class="suggestion-box suggestion-hero"',
    1,
)
html = html[:suggestion.start()] + html[suggestion.end():]
main_marker = '<main><section class="stage">'
if main_marker not in html:
    raise SystemExit('main/stage marker not found')
html = html.replace(
    main_marker,
    '<main>\n' + suggestion_block + '\n<section class="stage">',
    1,
)

play = re.search(
    r'(<button id="stagePlay" class="stage-play".*?</button>)',
    html,
    re.S,
)
if not play:
    raise SystemExit('stage play button not found')
stage_controls = (
    '<div class="stage-actions" aria-label="Навигация по actions">\n'
    '<button id="stagePrevAction" class="stage-nav-button" type="button" title="Предыдущий action" aria-label="Предыдущий action">‹</button>\n'
    + play.group(1)
    + '\n<button id="stageNextAction" class="stage-nav-button" type="button" title="Следующий action" aria-label="Следующий action">›</button>\n'
    '</div>'
)
html = html[:play.start()] + stage_controls + html[play.end():]
html_path.write_text(html)

js = js_path.read_text()
camera_marker = "for (const button of document.querySelectorAll('[data-view]'))"
if camera_marker not in js:
    raise SystemExit('camera handler marker not found')
js = js.replace(
    camera_marker,
    "$('stagePrevAction').onclick = () => $('prevAction').click();\n"
    "$('stageNextAction').onclick = () => $('nextAction').click();\n"
    + camera_marker,
    1,
)
js_path.write_text(js)

css = css_path.read_text()
if 'Primary contribution CTA' in css:
    raise SystemExit('UX additions already present')
additions = r'''

/* Primary contribution CTA: this is intentionally the first content block. */
.suggestion-hero{grid-column:1/-1;display:grid;grid-template-columns:minmax(230px,310px) minmax(0,1fr);gap:18px 28px;align-items:center;padding:18px 24px;background:linear-gradient(135deg,#0d1722 0%,#121f2d 72%,#112431 100%);border-bottom:1px solid #344257;box-shadow:0 8px 30px #0003;position:relative;z-index:7}
.suggestion-hero::before{content:'';position:absolute;inset:0 auto 0 0;width:4px;background:#79d8c8}
.suggestion-hero h2{font-size:24px;margin:7px 0 5px;letter-spacing:-.025em}
.suggestion-hero .suggestion-help{margin:0;font-size:14px;max-width:40ch}
.suggestion-hero form{display:grid;grid-template-columns:minmax(300px,1fr) minmax(220px,320px);grid-template-rows:auto auto auto;gap:7px 14px;align-items:center}
.suggestion-hero textarea{grid-column:1;grid-row:1/4;min-height:94px;max-height:150px;resize:vertical;font-size:16px;border-color:#496078;background:#16263a;box-shadow:inset 0 1px 0 #ffffff08}
.suggestion-hero textarea:focus{border-color:#8ce0d1;box-shadow:0 0 0 3px #77dfce25,inset 0 1px 0 #ffffff08}
.suggestion-hero .suggestion-example{grid-column:2;grid-row:1}
.suggestion-hero .suggestion-footer{grid-column:2;grid-row:2}
.suggestion-hero .suggestion-status{grid-column:2;grid-row:3;min-height:18px}
.suggestion-hero .suggestion-footer button{min-width:154px}

/* Browse adjacent motions directly around the stage play/pause control. */
.stage-actions{position:absolute;top:24px;right:18px;z-index:6;display:flex;align-items:center;gap:6px;pointer-events:auto}
.stage-actions .stage-play{position:static;top:auto;right:auto}
.stage-nav-button{width:42px;height:54px;border-radius:999px;padding:0;display:grid;place-items:center;background:#14202dd9;border:1px solid #52647a;color:#dbe8f3;font-size:30px;line-height:1;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);box-shadow:0 8px 24px #0005;transition:transform .16s ease,background .16s ease,border-color .16s ease}
.stage-nav-button:hover{background:#203449;border-color:#8ce0d1;transform:scale(1.04)}
.stage-nav-button:active{transform:scale(.96)}
.stage-title{padding-right:190px}

@media(max-width:900px){
  .suggestion-hero{grid-template-columns:1fr;padding:16px 18px;gap:10px}
  .suggestion-hero form{grid-template-columns:minmax(0,1fr) minmax(210px,280px)}
  .suggestion-hero textarea{min-height:88px}
}
@media(max-width:720px){
  .suggestion-hero{display:grid;grid-template-columns:1fr;gap:10px;padding:16px 14px 17px}
  .suggestion-hero::before{width:3px}
  .suggestion-hero h2{font-size:22px;margin-top:6px}
  .suggestion-hero .suggestion-help{font-size:13px}
  .suggestion-hero form{grid-template-columns:1fr;grid-template-rows:auto;gap:8px}
  .suggestion-hero textarea,.suggestion-hero .suggestion-example,.suggestion-hero .suggestion-footer,.suggestion-hero .suggestion-status{grid-column:1;grid-row:auto}
  .suggestion-hero textarea{min-height:108px;max-height:180px;font-size:16px}
  .suggestion-hero .suggestion-footer{display:grid;grid-template-columns:auto 1fr;align-items:center}
  .suggestion-hero .suggestion-footer button{width:100%;min-height:48px}
  .stage-actions{top:11px;right:8px;gap:4px}
  .stage-actions .stage-play{width:52px;height:52px}
  .stage-nav-button{width:34px;height:48px;font-size:26px}
  .stage-title{padding-right:142px}
}
@media(max-width:380px){
  .suggestion-hero{padding-inline:12px}
  .stage-actions{right:6px;gap:3px}
  .stage-actions .stage-play{width:48px;height:48px}
  .stage-nav-button{width:31px;height:44px;font-size:24px}
  .stage-title{padding-right:132px}
}
'''
css_path.write_text(css + additions)
