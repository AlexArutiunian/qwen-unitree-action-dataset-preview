from pathlib import Path

root = Path('.')
main_path = root / 'web/src/main.js'
worker_path = root / 'web/src/engine.worker.js'
html_path = root / 'web/index.html'
css_path = root / 'web/src/style.css'

main = main_path.read_text()
worker = worker_path.read_text()
html = html_path.read_text()
css = css_path.read_text()

# ---- Search: numeric IDs should work naturally ----
old = '''function matchesAction(action, query) {\n  if (!query) return true;\n  const haystack = `${action.sample_id} ${action.source_id} ${action.split} ${action.class} ${action.augmentation_type} ${action.text}`.toLocaleLowerCase('ru');\n  return haystack.includes(query);\n}\n'''
new = '''function matchesAction(action, query) {\n  if (!query) return true;\n  const normalized = query.trim().toLocaleLowerCase('ru');\n  if (/^\\d+$/.test(normalized)) {\n    const id = Number(normalized);\n    return Number(action.sample_id) === id || Number(action.source_id) === id;\n  }\n  const haystack = `${action.sample_id} ${action.source_id} ${action.split} ${action.class} ${action.augmentation_type} ${action.text}`.toLocaleLowerCase('ru');\n  return haystack.includes(normalized);\n}\n'''
assert old in main, 'matchesAction block not found'
main = main.replace(old, new, 1)

old = '''  if (visible.some(action => action.sample_id === currentAction.sample_id)) select.value = String(currentAction.sample_id);\n  else select.selectedIndex = -1;\n  $('datasetCount').textContent = `${visible.length} / ${actions.length}`;\n}\n'''
new = '''  if (visible.some(action => action.sample_id === currentAction.sample_id)) select.value = String(currentAction.sample_id);\n  else if (visible.length) select.selectedIndex = 0;\n  else select.selectedIndex = -1;\n  $('datasetCount').textContent = `${visible.length} / ${actions.length}`;\n}\n'''
assert old in main, 'populate selection block not found'
main = main.replace(old, new, 1)

old = "$('actionSearch').oninput = populateActions; $('splitFilter').onchange = populateActions;\n"
new = """$('actionSearch').oninput = populateActions; $('splitFilter').onchange = populateActions;\n$('actionSearch').addEventListener('keydown', event => {\n  if (event.key !== 'Enter') return;\n  event.preventDefault();\n  const first = $('actionSelect').selectedOptions[0];\n  if (!first) return;\n  const action = actions.find(item => item.sample_id === Number(first.value));\n  if (action) selectAction(action);\n});\n"""
assert old in main, 'search binding not found'
main = main.replace(old, new, 1)

# ---- Manual pose state ----
old = "let renderer, scene, camera, controls, meshes = [], bounds, initialValues, software = false, animationId, dirty = true, headlight;\n"
new = """let renderer, scene, camera, controls, meshes = [], bounds, initialValues, software = false, animationId, dirty = true, headlight;\nlet manualJointSpecs = [], manualPoseValues = {}, lastPoseValues = {}, manualPoseActive = false;\n"""
assert old in main, 'state declaration not found'
main = main.replace(old, new, 1)

insert_after = '''function jointLabel(name) {\n  if (labels[name]) return labels[name];\n  return name.replace(/_joint$/, '').replaceAll('_', ' ');\n}\n'''
manual_code = r'''function buildManualPoseControls(specs = []) {
  manualJointSpecs = specs.filter(spec => labels[spec.name]);
  const host = $('manualPoseControls');
  if (!host) return;
  host.replaceChildren();
  manualPoseValues = Object.fromEntries(manualJointSpecs.map(spec => [spec.name, spec.initial]));
  for (const spec of manualJointSpecs) {
    const row = document.createElement('label'); row.className = 'manual-joint-row';
    const name = document.createElement('span'); name.className = 'manual-joint-name'; name.textContent = jointLabel(spec.name); name.title = spec.name;
    const input = document.createElement('input'); input.type = 'range'; input.min = spec.min.toFixed(1); input.max = spec.max.toFixed(1); input.step = '0.5'; input.value = spec.initial.toFixed(1); input.dataset.joint = spec.name;
    const output = document.createElement('output'); output.textContent = `${spec.initial.toFixed(1)}°`; output.dataset.jointValue = spec.name;
    input.addEventListener('input', () => {
      manualPoseActive = true;
      playing = false; syncPlaybackUi();
      manualPoseValues[spec.name] = Number(input.value);
      output.textContent = `${Number(input.value).toFixed(1)}°`;
      if (modelReady) worker.postMessage({ type: 'manualPose', values: manualPoseValues, time: wanted, version: motionVersion });
    });
    row.append(name, input, output); host.append(row);
  }
  const empty = $('manualPoseEmpty');
  if (empty) empty.hidden = manualJointSpecs.length > 0;
}
function syncManualPoseControls(values) {
  if (!values || manualPoseActive) return;
  lastPoseValues = values;
  for (const spec of manualJointSpecs) {
    const value = Number(values[spec.name]);
    if (!Number.isFinite(value)) continue;
    manualPoseValues[spec.name] = value;
    const input = document.querySelector(`#manualPoseControls input[data-joint="${spec.name}"]`);
    const output = document.querySelector(`#manualPoseControls output[data-joint-value="${spec.name}"]`);
    if (input) input.value = String(Math.max(Number(input.min), Math.min(Number(input.max), value)));
    if (output) output.textContent = `${value.toFixed(1)}°`;
  }
}
function setManualNeutral() {
  if (!manualJointSpecs.length || !modelReady) return;
  manualPoseActive = true; playing = false; syncPlaybackUi();
  manualPoseValues = Object.fromEntries(manualJointSpecs.map(spec => [spec.name, spec.initial]));
  for (const spec of manualJointSpecs) {
    const input = document.querySelector(`#manualPoseControls input[data-joint="${spec.name}"]`);
    const output = document.querySelector(`#manualPoseControls output[data-joint-value="${spec.name}"]`);
    if (input) input.value = spec.initial.toFixed(1);
    if (output) output.textContent = `${spec.initial.toFixed(1)}°`;
  }
  worker.postMessage({ type: 'manualPose', values: manualPoseValues, time: wanted, version: motionVersion });
}
function restoreActionMotion() {
  if (!modelReady) return;
  manualPoseActive = false; playing = false; syncPlaybackUi();
  motionVersion += 1; ready = false; busy = false; wanted = 0; time = 0; duration = 0; initialValues = null;
  setControls(false); clock(); $('check').textContent = 'Возвращаем action…';
  worker.postMessage({ type: 'loadMotion', program: currentAction.program, version: motionVersion });
}
'''
assert insert_after in main, 'jointLabel block not found'
main = main.replace(insert_after, insert_after + manual_code, 1)

# reset manual mode whenever another action is selected
old = "  currentAction = action; motionVersion += 1; ready = false; busy = false; wanted = 0; time = 0; duration = 0; playbackDirection = 1; initialValues = null; unsupported = new Set();\n"
new = "  currentAction = action; motionVersion += 1; ready = false; busy = false; wanted = 0; time = 0; duration = 0; playbackDirection = 1; initialValues = null; unsupported = new Set(); manualPoseActive = false;\n"
assert old in main, 'selectAction state block not found'
main = main.replace(old, new, 1)

# manual control buttons
needle = "$('stageNextAction').onclick = () => $('nextAction').click();\n"
replacement = needle + "$('manualPoseNeutral')?.addEventListener('click', setManualNeutral);\n$('manualPoseRestore')?.addEventListener('click', restoreActionMotion);\n"
assert needle in main, 'stage nav binding not found'
main = main.replace(needle, replacement, 1)

# Receive joint ranges from worker when the model is ready.
needle = "      clearTimeout(timer); modelReady = true;\n"
replacement = "      clearTimeout(timer); modelReady = true; buildManualPoseControls(data.manualJoints || []);\n"
assert needle in main, 'ready block not found'
main = main.replace(needle, replacement, 1)

# Manual poses reuse the normal transform rendering path, but skip target-error text.
needle = "      if (!initialValues) initialValues = data.values;\n"
replacement = "      if (!initialValues) initialValues = data.values;\n      if (data.manual) { lastPoseValues = data.values; manualPoseValues = { ...manualPoseValues, ...Object.fromEntries(manualJointSpecs.map(spec => [spec.name, data.values[spec.name]]).filter(([, value]) => Number.isFinite(value))) }; }\n      else syncManualPoseControls(data.values);\n"
assert needle in main, 'pose values block not found'
main = main.replace(needle, replacement, 1)

old = """      const supportedTargets = targets.filter(joint => Number.isFinite(data.values[joint.name]));\n      const targetNames = new Set(supportedTargets.map(joint => joint.name));\n      const neutral = Object.entries(data.values).every(([name, value]) => targetNames.has(name) || Math.abs(value - initialValues[name]) < 1e-8);\n      const errors = supportedTargets.map(joint => Math.abs(data.values[joint.name] - joint.angle));\n      const error = errors.length ? Math.max(...errors) : 0;\n      const handNote = unsupported.size ? ` · ${unsupported.size} Inspire Hand вне web-модели` : '';\n      $('check').textContent = data.time >= duration ? `Цель: ошибка ${error.toFixed(6)}° · Остальные суставы ${neutral ? 'не изменены' : 'ИЗМЕНЕНЫ'}${handNote}` : `Плавная интерполяция · Остальные суставы в qpos0${handNote}`;\n"""
new = """      if (data.manual) {\n        $('check').textContent = 'Ручная поза · двигаются только руки и пояс · ноги в нейтрали';\n      } else {\n        const supportedTargets = targets.filter(joint => Number.isFinite(data.values[joint.name]));\n        const targetNames = new Set(supportedTargets.map(joint => joint.name));\n        const neutral = Object.entries(data.values).every(([name, value]) => targetNames.has(name) || Math.abs(value - initialValues[name]) < 1e-8);\n        const errors = supportedTargets.map(joint => Math.abs(data.values[joint.name] - joint.angle));\n        const error = errors.length ? Math.max(...errors) : 0;\n        const handNote = unsupported.size ? ` · ${unsupported.size} Inspire Hand вне web-модели` : '';\n        $('check').textContent = data.time >= duration ? `Цель: ошибка ${error.toFixed(6)}° · Остальные суставы ${neutral ? 'не изменены' : 'ИЗМЕНЕНЫ'}${handNote}` : `Плавная интерполяция · Остальные суставы в qpos0${handNote}`;\n      }\n"""
assert old in main, 'pose status block not found'
main = main.replace(old, new, 1)

# ---- Worker: safe direct pose editing with real MuJoCo joint limits ----
old = "let mj, model, data, timeline, joints, visible, initial;\n"
new = """let mj, model, data, timeline, joints, visible, initial;\nconst MANUAL_JOINTS = new Set([\n  'waist_yaw_joint','waist_roll_joint','waist_pitch_joint',\n  'left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint',\n  'right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint'\n]);\n"""
assert old in worker, 'worker state not found'
worker = worker.replace(old, new, 1)

# Give joint entries their model id so limits can be read.
old = "      if (name) joints.push({ name, address: model.jnt_qposadr[id] });\n"
new = "      if (name) joints.push({ name, address: model.jnt_qposadr[id], id });\n"
assert old in worker, 'joint collection line not found'
worker = worker.replace(old, new, 1)

insert_before = "function loadMotion(program, version) {\n"
manual_worker = r'''function manualJointSpecs() {
  return joints.filter(joint => MANUAL_JOINTS.has(joint.name)).map(joint => {
    const limited = Boolean(model.jnt_limited[joint.id]);
    const min = limited ? model.jnt_range[joint.id * 2] : -Math.PI;
    const max = limited ? model.jnt_range[joint.id * 2 + 1] : Math.PI;
    return { name: joint.name, min: min * 180 / Math.PI, max: max * 180 / Math.PI, initial: model.qpos0[joint.address] * 180 / Math.PI };
  });
}
function sendManualPose(values = {}, time = 0, version = currentVersion) {
  data.qpos.set(model.qpos0);
  for (const joint of joints) {
    if (!MANUAL_JOINTS.has(joint.name)) continue;
    const raw = Number(values[joint.name]);
    if (!Number.isFinite(raw)) continue;
    const limited = Boolean(model.jnt_limited[joint.id]);
    const min = limited ? model.jnt_range[joint.id * 2] : -Math.PI;
    const max = limited ? model.jnt_range[joint.id * 2 + 1] : Math.PI;
    const radians = Math.max(min, Math.min(max, raw * Math.PI / 180));
    data.qpos[joint.address] = radians;
  }
  mj.mj_forward(model, data);
  const transforms = new Float64Array(visible.length * 12);
  for (let i = 0; i < visible.length; i++) {
    const id = visible[i]; transforms.set(data.geom_xpos.subarray(id * 3, id * 3 + 3), i * 12);
    transforms.set(data.geom_xmat.subarray(id * 9, id * 9 + 9), i * 12 + 3);
  }
  const poseValues = Object.fromEntries(joints.map(joint => [joint.name, data.qpos[joint.address] * 180 / Math.PI]));
  postMessage({ type: 'pose', time, version, transforms, values: poseValues, manual: true }, [transforms.buffer]);
}
'''
assert insert_before in worker, 'loadMotion not found'
worker = worker.replace(insert_before, manual_worker + insert_before, 1)

needle = "    if (message.type === 'loadMotion') {\n"
replacement = """    if (message.type === 'manualPose') {\n      if (data && model && !initializing && message.version === currentVersion) sendManualPose(message.values, message.time, message.version);\n      return;\n    }\n    if (message.type === 'loadMotion') {\n"""
assert needle in worker, 'worker onmessage loadMotion not found'
worker = worker.replace(needle, replacement, 1)

old = "    postMessage({ type: 'ready', geometries, triangles: triangleCount, format: mode, compileMs: performance.now() - start, nq: model.nq, njnt: model.njnt, version: message.version }, transfers);\n"
new = "    postMessage({ type: 'ready', geometries, triangles: triangleCount, format: mode, compileMs: performance.now() - start, nq: model.nq, njnt: model.njnt, manualJoints: manualJointSpecs(), version: message.version }, transfers);\n"
assert old in worker, 'ready post not found'
worker = worker.replace(old, new, 1)

# ---- HTML panel ----
old = '''<small id="datasetStatus">Загрузка полного датасета…</small>\n</div>\n<div class="divider"></div><span class="eyebrow">КОМАНДА</span>'''
new = '''<small id="datasetStatus">Загрузка полного датасета…</small>\n</div>\n<details class="manual-pose" id="manualPosePanel">\n<summary><span>Ручная поза</span><small>руки + пояс</small></summary>\n<p class="manual-pose-help">Двигай ползунки — поза меняется сразу в MuJoCo. Ноги остаются в нейтрали.</p>\n<p id="manualPoseEmpty" class="manual-pose-empty">Ползунки появятся после загрузки 3D-модели.</p>\n<div id="manualPoseControls" class="manual-pose-controls"></div>\n<div class="manual-pose-actions"><button id="manualPoseNeutral" type="button">Нейтраль</button><button id="manualPoseRestore" type="button">Вернуться к action</button></div>\n</details>\n<div class="divider"></div><span class="eyebrow">КОМАНДА</span>'''
assert old in html, 'aside insertion point not found'
html = html.replace(old, new, 1)

# Search affordance: make numeric IDs explicit without removing text search.
html = html.replace('placeholder="ID, команда, class…"', 'placeholder="ID / source_id / команда…"', 1)

# ---- CSS ----
marker = '/* Manual pose controls v1 */'
assert marker not in css, 'manual CSS already applied'
css += r'''

/* Manual pose controls v1 */
.manual-pose{margin-top:16px;padding:0;border:1px solid #314154;border-radius:10px;background:#131d29;overflow:hidden}
.manual-pose>summary{display:flex;align-items:center;gap:8px;padding:12px 13px;cursor:pointer;list-style:none;font-weight:700;color:#e8f2f7;background:#172332}
.manual-pose>summary::-webkit-details-marker{display:none}
.manual-pose>summary::after{content:'›';margin-left:auto;color:#83dccc;font-size:22px;line-height:1;transition:transform .15s ease}
.manual-pose[open]>summary::after{transform:rotate(90deg)}
.manual-pose>summary small{font-size:11px;font-weight:600;color:#8fa8bc;letter-spacing:.04em}
.manual-pose-help,.manual-pose-empty{margin:10px 12px 8px;font-size:12px;line-height:1.4;color:#91a6ba}
.manual-pose-controls{display:grid;gap:8px;padding:4px 12px 12px;max-height:46vh;overflow:auto}
.manual-joint-row{display:grid;grid-template-columns:minmax(108px,1fr) minmax(110px,1.45fr) 48px;align-items:center;gap:8px;font-size:12px}
.manual-joint-name{color:#c5d3df;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.manual-joint-row input[type=range]{width:100%;min-width:0;accent-color:#78d8c8}
.manual-joint-row output{font-variant-numeric:tabular-nums;text-align:right;color:#8de0d2;font-size:11px}
.manual-pose-actions{display:grid;grid-template-columns:1fr 1.3fr;gap:8px;padding:0 12px 12px}
.manual-pose-actions button{min-height:38px;padding:7px 9px;font-size:12px}
#manualPoseRestore{border-color:#5c756f;color:#a7e4d9}
@media(max-width:720px){
  .manual-pose{margin-top:14px}
  .manual-pose>summary{padding:13px 12px;font-size:15px}
  .manual-pose-controls{max-height:none;overflow:visible;padding:5px 10px 12px}
  .manual-joint-row{grid-template-columns:104px minmax(0,1fr) 45px;gap:7px;font-size:11.5px}
  .manual-joint-row input[type=range]{min-height:32px}
  .manual-pose-actions{padding:0 10px 12px}
  .manual-pose-actions button{min-height:42px}
}
'''

main_path.write_text(main)
worker_path.write_text(worker)
html_path.write_text(html)
css_path.write_text(css)
print('patched')
