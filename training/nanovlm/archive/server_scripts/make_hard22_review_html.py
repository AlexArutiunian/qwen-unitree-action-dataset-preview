from pathlib import Path
import pandas as pd
import html
import shutil

DATASET = Path("real_glass_hard22_yesno4_eval")
OUT = Path("hard22_review_models")
OUT.mkdir(exist_ok=True)

base = pd.read_csv(DATASET / "metadata.csv")

qwen_paths = {
    "Qwen2.5-VL-3B": "qwen25vl3b_hard22_yesno4/qwen_fourchoice_explain_results.csv",
    "Qwen2.5-VL-7B": "qwen25vl7b_hard22_yesno4/qwen_fourchoice_explain_results.csv",
    "Qwen3-VL-8B": "qwen3vl8b_hard22_yesno4/qwen3_results.csv",
}

nano_paths = {
    "nano raw": "eval_hard22_nano_raw/nanovlm_results.csv",
    "nano AI2THOR": "eval_hard22_nano_ai2thor_1k/nanovlm_results.csv",
    "nano affordance": "eval_hard22_nano_affordance/nanovlm_results.csv",
    "nano reflect": "eval_hard22_nano_reflect/nanovlm_results.csv",
}

for _, r in base.iterrows():
    src = DATASET / r["image"]
    dst = OUT / Path(r["image"]).name
    if src.exists():
        shutil.copy2(src, dst)

parts = ["""
<html><head><meta charset="utf-8">
<style>
body{font-family:Arial,sans-serif;margin:20px;background:#f7f7f7}
.card{background:white;border:1px solid #bbb;border-radius:10px;padding:16px;margin-bottom:24px}
img{max-width:760px;max-height:560px;display:block;margin:10px 0}
.ok{color:green;font-weight:bold}.bad{color:red;font-weight:bold}
table{border-collapse:collapse;width:100%;margin-top:10px}
td,th{border:1px solid #ccc;padding:6px;vertical-align:top}
.small{font-size:13px;color:#333}
</style></head><body>
<h1>hard22 review: GT vs Qwen/nano</h1>
"""]

for i, r in base.iterrows():
    img_name = Path(r["image"]).name
    parts.append(f'<div class="card"><h2>#{i}: {html.escape(str(r["image"]))}</h2>')
    parts.append(f'<img src="{img_name}">')
    parts.append(f'<p><b>Question:</b> {html.escape(str(r.get("question","")))}</p>')
    parts.append(f'<p><b>GT answer:</b> {html.escape(str(r.get("answer","")))} | gt_idx={html.escape(str(r.get("gt_idx","")))}</p>')
    parts.append(f'<p><b>Choices:</b> {html.escape(str(r.get("choices","")))}</p>')
    parts.append("<table><tr><th>Model</th><th>Pred</th><th>Correct</th><th>Explanation / probs</th></tr>")

    for name, path in qwen_paths.items():
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        qr = df.iloc[i]

        if "qwen_pred_answer" in qr:
            pred = qr["qwen_pred_answer"]
            ok = int(qr["qwen_agrees_with_dataset"]) == 1
            detail = str(qr.get("qwen_explanation", "")) + "<br><span class='small'>" + html.escape(str(qr.get("qwen_raw_output", ""))) + "</span>"
        else:
            pred = qr["qwen3_pred_answer"]
            ok = int(qr["qwen3_correct"]) == 1
            detail = "<span class='small'>" + html.escape(str(qr.get("qwen3_raw_output", ""))) + "</span>"

        parts.append(
            f'<tr><td>{html.escape(name)}</td><td>{html.escape(str(pred))}</td>'
            f'<td class="{"ok" if ok else "bad"}">{ok}</td><td>{detail}</td></tr>'
        )

    for name, path in nano_paths.items():
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        nr = df.iloc[i]
        ok = int(nr["nanovlm_correct"]) == 1
        detail = f'letter={nr["nanovlm_pred_letter"]}, probs=({nr["prob_A"]:.3f}, {nr["prob_B"]:.3f}, {nr["prob_C"]:.3f}, {nr["prob_D"]:.3f})'
        parts.append(
            f'<tr><td>{html.escape(name)}</td><td>{html.escape(str(nr["nanovlm_pred_answer"]))}</td>'
            f'<td class="{"ok" if ok else "bad"}">{ok}</td><td>{html.escape(detail)}</td></tr>'
        )

    parts.append("</table></div>")

parts.append("</body></html>")
(OUT / "index.html").write_text("\n".join(parts), encoding="utf-8")
print("saved:", OUT / "index.html")
