from pathlib import Path

p = Path('web/src/style.css')
css = p.read_text()
marker = '/* Compact contribution bar v2 */'
if marker in css:
    raise SystemExit('compact prompt CSS already present')
css += r'''

/* Compact contribution bar v2 */
.suggestion-hero{
  grid-column:1/-1;
  display:grid;
  grid-template-columns:auto minmax(0,1fr);
  grid-template-areas:'prompt-title prompt-form';
  align-items:center;
  gap:10px 16px;
  min-height:82px;
  padding:9px 18px 8px;
  background:#111b27;
  border-bottom:1px solid #344257;
  box-shadow:none;
  position:relative;
  z-index:7
}
.suggestion-hero::before{width:3px}
.suggestion-hero>.eyebrow,.suggestion-hero>.suggestion-help,.suggestion-hero .suggestion-example{display:none}
.suggestion-hero h2{grid-area:prompt-title;margin:0;font-size:18px;line-height:1.15;letter-spacing:-.02em;white-space:nowrap}
.suggestion-hero form{grid-area:prompt-form;display:grid;grid-template-columns:minmax(0,1fr) auto;grid-template-rows:auto auto;gap:4px 8px;align-items:center}
.suggestion-hero textarea{grid-column:1;grid-row:1;width:100%;height:46px;min-height:46px;max-height:72px;resize:vertical;padding:11px 12px;font-size:15px;line-height:1.35;background:#17283a;border-color:#42576f}
.suggestion-hero .suggestion-footer{grid-column:2;grid-row:1;display:block}
.suggestion-hero .suggestion-footer>span{display:none}
.suggestion-hero .suggestion-footer button{width:auto;min-width:142px;height:46px;min-height:46px;padding:0 15px;white-space:nowrap}
.suggestion-hero .suggestion-status{grid-column:1/-1;grid-row:2;min-height:13px;margin:0;font-size:10.5px;line-height:1.2;color:#8095aa}

@media(max-width:720px){
  .suggestion-hero{display:grid;grid-template-columns:1fr;grid-template-areas:'prompt-title' 'prompt-form';gap:6px;min-height:0;padding:8px 10px 8px}
  .suggestion-hero h2{font-size:16px;line-height:1.1;white-space:normal}
  .suggestion-hero form{display:grid;grid-template-columns:minmax(0,1fr) auto;grid-template-rows:auto auto;gap:3px 6px}
  .suggestion-hero textarea{grid-column:1;grid-row:1;height:44px;min-height:44px;max-height:72px;padding:10px 11px;font-size:14px;resize:none}
  .suggestion-hero .suggestion-footer{grid-column:2;grid-row:1;display:block}
  .suggestion-hero .suggestion-footer button{width:auto;min-width:92px;height:44px;min-height:44px;padding:0 11px;font-size:13px}
  .suggestion-hero .suggestion-status{grid-column:1/-1;grid-row:2;min-height:11px;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
}
@media(max-width:380px){
  .suggestion-hero{padding-inline:8px}
  .suggestion-hero h2{font-size:15px}
  .suggestion-hero textarea{font-size:13px;padding-inline:9px}
  .suggestion-hero .suggestion-footer button{min-width:82px;padding-inline:9px;font-size:12px}
}
'''
p.write_text(css)
