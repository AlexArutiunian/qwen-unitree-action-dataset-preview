import * as THREE from 'three';
import { SVGRenderer } from 'three/addons/renderers/SVGRenderer.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';
import { sample91 } from './sample.js';
import { SUGGESTION_ENDPOINT } from './submission-config.js';
import './style.css';

const $ = id => document.getElementById(id);
const HF_BASE = 'https://huggingface.co/datasets/AlexArutiunian/g1-json-to-action/resolve/main/data/';
const DATA_FILES = ['train.jsonl', 'val.jsonl', 'reserved.jsonl'];
const fallbackAction = { ...sample91, source_id: 13, augmentation_type: 'intensity_large' };
const labels = {
  waist_yaw_joint: 'Талия: yaw', waist_roll_joint: 'Талия: roll', waist_pitch_joint: 'Талия: pitch',
  left_shoulder_pitch_joint: 'Л. плечо: pitch', left_shoulder_roll_joint: 'Л. плечо: roll', left_shoulder_yaw_joint: 'Л. плечо: yaw', left_elbow_joint: 'Л. локоть',
  left_wrist_roll_joint: 'Л. запястье: roll', left_wrist_pitch_joint: 'Л. запястье: pitch', left_wrist_yaw_joint: 'Л. запястье: yaw',
  right_shoulder_pitch_joint: 'П. плечо: pitch', right_shoulder_roll_joint: 'П. плечо: roll', right_shoulder_yaw_joint: 'П. плечо: yaw', right_elbow_joint: 'П. локоть',
  right_wrist_roll_joint: 'П. запястье: roll', right_wrist_pitch_joint: 'П. запястье: pitch', right_wrist_yaw_joint: 'П. запястье: yaw'
};

let actions = [fallbackAction], currentAction = fallbackAction, targets = [], cells = new Map(), unsupported = new Set();
let worker, timer, playing = false, playbackDirection = 1, time = 0, duration = 1.5, ready = false, modelReady = false, busy = false, wanted = 0, motionVersion = 1;
let renderer, scene, camera, controls, meshes = [], bounds, initialValues, software = false, animationId, dirty = true, headlight;
const matrix = new THREE.Matrix4(), center = new THREE.Vector3(), direction = new THREE.Vector3(3, -2, 0.9), headlightDirection = new THREE.Vector3();

function padId(id) { return String(id).padStart(5, '0'); }
function normalizeCommand(text = '') { return text.replace(/^\s*Command\s*:\s*/i, '').trim(); }
function actionTitle(action) {
  const text = action.text.replace(/\s+/g, ' ').trim();
  return `#${padId(action.sample_id)} · ${action.split} · ${action.augmentation_type || 'original'} · ${text}`;
}
function normalizeRow(row) {
  const user = row.messages?.find(message => message.role === 'user');
  const assistant = row.messages?.find(message => message.role === 'assistant');
  if (!user || !assistant) throw new Error(`sample ${row.sample_id}: missing user/assistant message`);
  const program = JSON.parse(assistant.content);
  if (!Array.isArray(program)) throw new Error(`sample ${row.sample_id}: motion JSON is not an array`);
  return {
    sample_id: row.sample_id, source_id: row.source_id, class: row.class,
    augmentation_type: row.augmentation_type, split: row.split,
    text: normalizeCommand(user.content), program
  };
}
function collectTargets(program) {
  const order = [], final = new Map();
  const walk = blocks => {
    for (const block of blocks || []) {
      if (Array.isArray(block.repeat)) {
        if ((block.times ?? 0) > 0) walk(block.repeat);
        continue;
      }
      for (const joint of block.frame || []) {
        if (!final.has(joint.name)) order.push(joint.name);
        final.set(joint.name, joint.angle);
      }
    }
  };
  walk(program);
  return order.map(name => ({ name, angle: final.get(name) }));
}
function jointLabel(name) {
  if (labels[name]) return labels[name];
  return name.replace(/_joint$/, '').replaceAll('_', ' ');
}
function rebuildJointTable() {
  targets = collectTargets(currentAction.program);
  cells = new Map();
  $('joints').replaceChildren();
  if (!targets.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td'); cell.colSpan = 3; cell.textContent = 'Пауза / удержание позы';
    row.append(cell); $('joints').append(row); return;
  }
  for (const joint of targets) {
    const row = document.createElement('tr');
    const name = document.createElement('td'); name.textContent = jointLabel(joint.name); name.title = joint.name;
    const actual = document.createElement('td'); actual.textContent = '—'; actual.dataset.joint = joint.name;
    const goal = document.createElement('td'); goal.textContent = Number(joint.angle).toFixed(1);
    row.append(name, actual, goal); $('joints').append(row); cells.set(joint.name, { actual, row });
  }
}
function renderActionInfo() {
  $('sampleLabel').textContent = `SAMPLE ${padId(currentAction.sample_id)}`;
  $('stageCommand').textContent = currentAction.text || `Action #${currentAction.sample_id}`;
  $('stageMeta').textContent = `${currentAction.split} · ${currentAction.class} · ${currentAction.augmentation_type || 'original'}`;
  $('command').textContent = currentAction.text;
  $('json').textContent = JSON.stringify(currentAction.program, null, 2);
  document.title = `G1 · Action ${currentAction.sample_id} · MuJoCo`;
  $('referenceImage').hidden = currentAction.sample_id !== 91;
  $('referenceLink').hidden = currentAction.sample_id !== 91;
  rebuildJointTable();
}
function syncPlaybackUi() {
  const label = playing ? 'Пауза' : 'Воспроизвести';
  $('play').textContent = label;
  $('stagePlay').classList.toggle('is-playing', playing);
  $('stagePlay').setAttribute('aria-label', label);
  $('stagePlay').setAttribute('title', label);
  $('stagePlay').setAttribute('aria-pressed', String(playing));
}
function setControls(enabled) {
  $('play').disabled = !enabled;
  $('stagePlay').disabled = !enabled;
  $('reset').disabled = !enabled;
  $('seek').disabled = !enabled;
}
function fail(message) {
  ready = false; modelReady = false; playing = false; clearTimeout(timer); worker?.terminate();
  syncPlaybackUi(); setControls(false);
  $('fallback').hidden = false; $('loading').textContent = message; $('retry').hidden = false; $('retry').onclick = start;
  $('status').textContent = '3D недоступен';
}
function clock() { $('clock').textContent = `${time.toFixed(2)} / ${duration.toFixed(2)} с`; $('seek').value = time; }
function requestPose(value) {
  time = Math.max(0, Math.min(duration, value)); wanted = time; clock();
  if (!ready || busy) return;
  busy = true; worker.postMessage({ type: 'seek', time: wanted, version: motionVersion });
}
function fit(view) {
  if (!bounds) return;
  if (view === 'front') direction.set(1, 0, 0.22);
  if (view === 'side') direction.set(0, -1, 0.16);
  if (view === 'back') direction.set(-1, 0, 0.22);
  if (view === 'fit') direction.copy(camera.position).sub(controls.target);
  const size = bounds.getSize(new THREE.Vector3()); bounds.getCenter(center);
  const radius = Math.max(0.25, size.length() / 2);
  const halfV = THREE.MathUtils.degToRad(camera.fov / 2), halfH = Math.atan(Math.tan(halfV) * camera.aspect);
  const framing = camera.aspect < 0.82 ? 1.03 : 1.00;
  const distance = radius / Math.sin(Math.min(halfV, halfH)) * framing;
  camera.position.copy(center).addScaledVector(direction.normalize(), distance);
  camera.near = .01; camera.far = Math.max(100, distance * 10); camera.updateProjectionMatrix();
  controls.target.copy(center); controls.minDistance = radius * .5; controls.maxDistance = distance * 5; controls.update();
  dirty = true;
}
function toggle() {
  if (!ready || duration <= 0) return;
  playing = !playing;
  syncPlaybackUi();
}
function reset() {
  playing = false;
  playbackDirection = 1;
  syncPlaybackUi();
  requestPose(0);
}
function pingPongTime(delta) {
  if (!Number.isFinite(delta) || delta <= 0 || duration <= 0) return time;
  const cycle = duration * 2;
  let phase = playbackDirection > 0 ? time : cycle - time;
  phase = (phase + delta) % cycle;
  if (phase < duration) {
    playbackDirection = 1;
    return phase;
  }
  if (phase > duration) {
    playbackDirection = -1;
    return cycle - phase;
  }
  playbackDirection = -1;
  return duration;
}

function selectAction(action, updateUrl = true) {
  if (!action || action.sample_id === currentAction.sample_id && action.program === currentAction.program) return;
  currentAction = action; motionVersion += 1; ready = false; busy = false; wanted = 0; time = 0; duration = 0; playbackDirection = 1; initialValues = null; unsupported = new Set();
  playing = false; syncPlaybackUi(); setControls(false); renderActionInfo(); clock();
  $('check').textContent = 'Подготовка motion…'; $('status').textContent = `Action #${currentAction.sample_id}`;
  if (updateUrl) {
    const url = new URL(location.href); url.searchParams.set('action', currentAction.sample_id); history.replaceState(null, '', url);
  }
  if (modelReady) worker.postMessage({ type: 'loadMotion', program: currentAction.program, version: motionVersion });
}
function matchesAction(action, query) {
  if (!query) return true;
  const haystack = `${action.sample_id} ${action.source_id} ${action.split} ${action.class} ${action.augmentation_type} ${action.text}`.toLocaleLowerCase('ru');
  return haystack.includes(query);
}
function sourceBrowseKey(action) {
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
function populateActions() {
  const query = $('actionSearch').value.trim().toLocaleLowerCase('ru');
  const split = $('splitFilter').value;
  const visible = actions.filter(action => (!split || action.split === split) && matchesAction(action, query));
  const select = $('actionSelect'); select.replaceChildren();
  for (const action of visible) {
    const option = document.createElement('option'); option.value = String(action.sample_id); option.textContent = actionTitle(action); select.append(option);
  }
  if (visible.some(action => action.sample_id === currentAction.sample_id)) select.value = String(currentAction.sample_id);
  else select.selectedIndex = -1;
  $('datasetCount').textContent = `${visible.length} / ${actions.length}`;
}
async function fetchDatasetFile(name) {
  const urls = [
    `${HF_BASE}${name}`,
    `https://huggingface.co/datasets/AlexArutiunian/g1-json-to-action/raw/main/data/${name}`
  ];
  let lastError;
  for (const url of urls) {
    try {
      const response = await fetch(url, { cache: 'force-cache' });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return await response.text();
    } catch (error) { lastError = error; }
  }
  throw lastError || new Error(`Не удалось загрузить ${name}`);
}
async function loadDataset() {
  $('datasetStatus').textContent = 'Загрузка полного датасета с Hugging Face…';
  try {
    const texts = await Promise.all(DATA_FILES.map(fetchDatasetFile));
    const loaded = [];
    for (const text of texts) {
      for (const line of text.split(/\r?\n/)) {
        if (!line.trim()) continue;
        loaded.push(normalizeRow(JSON.parse(line)));
      }
    }
    const unique = new Map(loaded.map(action => [action.sample_id, action]));
    actions = [...unique.values()].sort((a, b) => a.sample_id - b.sample_id);
    $('actionSelect').disabled = false; $('actionSearch').disabled = false; $('splitFilter').disabled = false;
    $('datasetStatus').textContent = `Загружено ${sourceRepresentatives().length} source_id · ${actions.length} actions`;
    populateActions();
    const requested = Number(new URLSearchParams(location.search).get('action')) || 91;
    const action = unique.get(requested) || unique.get(91) || actions[0];
    if (action) { selectAction(action, false); $('actionSelect').value = String(action.sample_id); }
  } catch (error) {
    $('datasetStatus').textContent = `Полный список недоступен: ${error.message}. Оставлен sample 91.`;
    $('actionSelect').disabled = true; $('actionSearch').disabled = true; $('splitFilter').disabled = true;
  }
}

function syncSuggestionCount() {
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
    status.textContent = '';
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
    submit.textContent = 'Отправить';
  }
}
$('suggestionForm')?.addEventListener('submit', submitSuggestion);
initSuggestionForm();

$('play').onclick = toggle;
$('stagePlay').onclick = toggle;
$('reset').onclick = reset;
$('seek').oninput = event => {
  playing = false;
  const value = Number(event.target.value);
  if (value <= 0) playbackDirection = 1;
  else if (value >= duration) playbackDirection = -1;
  syncPlaybackUi();
  requestPose(value);
};
$('actionSearch').oninput = populateActions; $('splitFilter').onchange = populateActions;
$('actionSelect').onchange = event => selectAction(actions.find(action => action.sample_id === Number(event.target.value)));
$('prevAction').onclick = () => browseSource(-1);
$('nextAction').onclick = () => browseSource(1);
$('stagePrevAction').onclick = () => $('prevAction').click();
$('stageNextAction').onclick = () => $('nextAction').click();
for (const button of document.querySelectorAll('[data-view]')) button.onclick = () => fit(button.dataset.view);
document.addEventListener('keydown', event => {
  if (/INPUT|SELECT|TEXTAREA|BUTTON|SUMMARY/.test(event.target.tagName) || event.target.isContentEditable) return;
  if (event.code === 'Space') { event.preventDefault(); toggle(); }
  if (event.code === 'KeyR') reset();
  if (event.code === 'ArrowLeft' || event.code === 'ArrowRight') {
    event.preventDefault(); playing = false; syncPlaybackUi(); requestPose(time + (event.code === 'ArrowLeft' ? -.1 : .1));
  }
});

function start() {
  worker?.terminate(); clearTimeout(timer); ready = false; modelReady = false; busy = false; initialValues = null; playing = false; playbackDirection = 1; time = 0; wanted = 0; syncPlaybackUi(); setControls(false); clock();
  $('retry').hidden = true; $('fallback').hidden = false; $('loading').textContent = 'Загрузка интерактивной модели…';
  for (const mesh of meshes) { scene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); } meshes = [];
  worker = new Worker(new URL('./engine.worker.js', import.meta.url), { type: 'module' });
  timer = setTimeout(() => fail('Загрузка заняла больше 60 секунд. Проверьте соединение и повторите попытку.'), 60000);
  worker.onerror = event => fail(event.message || 'Ошибка запуска MuJoCo');
  worker.onmessage = ({ data }) => {
    if (data.type === 'status') { $('status').textContent = data.text; $('loading').textContent = data.text; }
    if (data.type === 'error') fail(data.text);
    if (data.type === 'ready') {
      clearTimeout(timer); modelReady = true;
      meshes = data.geometries.map(g => {
        const indexed = new THREE.BufferGeometry();
        indexed.setAttribute('position', new THREE.Float32BufferAttribute(g.vertices, 3));
        indexed.setIndex(new THREE.BufferAttribute(g.faces, 1));

        // The official G1 geometry stays bit-for-bit at the same vertex
        // positions. Only shading normals are generated. Adjacent triangles
        // are visually blended below 50 degrees, while real mechanical edges
        // remain sharp. This removes the faceted/triangular look without
        // rounding or simplifying the robot body.
        let geometry = indexed;
        if (software) {
          geometry.computeVertexNormals();
        } else {
          geometry = toCreasedNormals(indexed, THREE.MathUtils.degToRad(50));
          if (geometry !== indexed) indexed.dispose();
        }
        geometry.computeBoundingSphere();

        // WBC-style material mapping. Keep MuJoCo's authored two-tone
        // palette, but bias it slightly darker/cooler so the 0.7 shell reads
        // as technical silver-gray instead of white plastic.
        const luma = 0.2126 * g.rgba[0] + 0.7152 * g.rgba[1] + 0.0722 * g.rgba[2];
        const grayScale = luma > .35 ? .88 : .80;
        const color = new THREE.Color().setRGB(
          Math.min(1, g.rgba[0] * grayScale),
          Math.min(1, g.rgba[1] * grayScale),
          Math.min(1, g.rgba[2] * grayScale),
          THREE.SRGBColorSpace
        );
        const shininess = luma > .35 ? .68 : .48;
        const material = software
          ? new THREE.MeshLambertMaterial({ color })
          : new THREE.MeshPhysicalMaterial({
              color,
              roughness: 1 - shininess,
              metalness: 0,
              specularIntensity: .46,
              specularColor: new THREE.Color(0xd4e4f0),
              flatShading: false,
              transparent: g.rgba[3] < .999,
              opacity: g.rgba[3]
            });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.matrixAutoUpdate = false;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
        return mesh;
      });
      const triangles = Number(data.triangles || 0).toLocaleString('ru-RU');
      $('metrics').textContent = `${data.format.toUpperCase()} · exact official mesh · ${triangles} граней · smooth crease 50° · WBC gray · ${data.compileMs.toFixed(0)} мс · ${meshes.length} деталей · nq=${data.nq}`;
      if (data.version !== motionVersion) worker.postMessage({ type: 'loadMotion', program: currentAction.program, version: motionVersion });
    }
    if (data.type === 'motionReady') {
      if (data.version !== motionVersion) return;
      duration = data.duration; $('seek').max = Math.max(duration, .001); time = 0; wanted = 0; playbackDirection = 1; busy = false; initialValues = null; unsupported = new Set(data.unsupported || []);
      bounds = new THREE.Box3(new THREE.Vector3(...data.bounds.min), new THREE.Vector3(...data.bounds.max)); fit('front');
      for (const [name, cell] of cells) cell.row.classList.toggle('unsupported', unsupported.has(name));
      ready = true; setControls(true); playing = duration > 0; syncPlaybackUi(); clock();
      $('stageMeta').textContent = `${currentAction.split} · ${currentAction.class} · ${currentAction.augmentation_type || 'original'} · ${duration.toFixed(2)} с · ping-pong`;
      $('status').textContent = software ? `#${currentAction.sample_id} · Программный 3D` : `#${currentAction.sample_id} · WebGL`;
      if (unsupported.size) $('check').textContent = `${unsupported.size} сустав(а) Inspire Hand не визуализируются этой web-моделью G1`;
    }
    if (data.type === 'pose') {
      if (data.version !== motionVersion) return;
      dirty = true;
      const t = data.transforms;
      for (let i = 0; i < meshes.length; i++) {
        const p = i * 12;
        matrix.set(t[p + 3], t[p + 4], t[p + 5], t[p], t[p + 6], t[p + 7], t[p + 8], t[p + 1], t[p + 9], t[p + 10], t[p + 11], t[p + 2], 0, 0, 0, 1);
        meshes[i].matrix.copy(matrix); meshes[i].matrixWorldNeedsUpdate = true;
      }
      if (!initialValues) initialValues = data.values;
      for (const joint of targets) {
        const cell = cells.get(joint.name); if (!cell) continue;
        const value = data.values[joint.name]; cell.actual.textContent = Number.isFinite(value) ? value.toFixed(1) : '—';
      }
      const supportedTargets = targets.filter(joint => Number.isFinite(data.values[joint.name]));
      const targetNames = new Set(supportedTargets.map(joint => joint.name));
      const neutral = Object.entries(data.values).every(([name, value]) => targetNames.has(name) || Math.abs(value - initialValues[name]) < 1e-8);
      const errors = supportedTargets.map(joint => Math.abs(data.values[joint.name] - joint.angle));
      const error = errors.length ? Math.max(...errors) : 0;
      const handNote = unsupported.size ? ` · ${unsupported.size} Inspire Hand вне web-модели` : '';
      $('check').textContent = data.time >= duration ? `Цель: ошибка ${error.toFixed(6)}° · Остальные суставы ${neutral ? 'не изменены' : 'ИЗМЕНЕНЫ'}${handNote}` : `Плавная интерполяция · Остальные суставы в qpos0${handNote}`;
      $('fallback').hidden = true; busy = false;
      if (Math.abs(wanted - data.time) > 1e-6) requestPose(wanted);
    }
  };
  worker.postMessage({ type: 'init', base: new URL('./', document.baseURI).href, format: new URLSearchParams(location.search).get('format'), lite: software, program: currentAction.program, version: motionVersion });
}

renderActionInfo(); populateActions(); syncPlaybackUi();
try {
  scene = new THREE.Scene(); scene.fog = new THREE.Fog('#16283a', 20, 60);
  camera = new THREE.PerspectiveCamera(36, 1, .01, 100); camera.up.set(0, 0, 1);
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('webgl2', { antialias: true, alpha: true });
  if (context) { renderer = new THREE.WebGLRenderer({ canvas, context, antialias: true, alpha: true }); renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap; renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.NoToneMapping; }
  else { software = true; renderer = new SVGRenderer(); renderer.setQuality('low'); renderer.setClearColor(0x17212e); renderer.domElement.style.width = '100%'; renderer.domElement.style.height = '100%'; }
  $('viewport').append(renderer.domElement);
  renderer.domElement.addEventListener('webglcontextlost', e => { e.preventDefault(); fail('WebGL-контекст потерян. Обновите страницу.'); });
  controls = new OrbitControls(camera, renderer.domElement); controls.addEventListener('change', () => { dirty = true; }); controls.enableDamping = true; controls.dampingFactor = .08;
  // Lighting follows the WBC viewer: low ambient, an overhead MuJoCo-like
  // key, cool floor bounce and a soft camera headlight. No HDR environment
  // and no filmic tone mapping, so gray body panels keep their separation.
  scene.add(new THREE.AmbientLight(0xe8eef4, software ? .18 : .24));
  scene.add(new THREE.HemisphereLight(0xd6e2f0, 0x3b5368, software ? .45 : .44));
  const light = new THREE.DirectionalLight(0xffffff, software ? .85 : 2.75);
  light.position.set(1.2, -.5, 5.6); light.target.position.set(0, 0, .9); scene.add(light.target);
  light.castShadow = true; light.shadow.mapSize.set(2048, 2048); light.shadow.camera.left = -3; light.shadow.camera.right = 3; light.shadow.camera.top = 3; light.shadow.camera.bottom = -3; light.shadow.bias = -.0002; light.shadow.normalBias = .02; scene.add(light);
  headlight = new THREE.DirectionalLight(0xe4eef8, software ? .25 : 1.12); headlight.castShadow = false; scene.add(headlight.target); scene.add(headlight);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: 0x2a455c, roughness: .96, metalness: 0 })); floor.position.z = -.007; floor.receiveShadow = true; if (!software) scene.add(floor);
  const grid = new THREE.GridHelper(12, 60, 0x429eb0, 0x2f7a8a); grid.rotation.x = Math.PI / 2; grid.position.z = -.005; if (software) { grid.material.vertexColors = false; grid.material.color.set(0x395668); } scene.add(grid);
  new ResizeObserver(() => {
    const { width, height } = $('viewport').getBoundingClientRect(); renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix(); fit('fit');
  }).observe($('viewport'));
  let last = performance.now();
  function animate(now) {
    const elapsed = Math.max(0, (now - last) / 1000); last = now;
    if (playing && ready && duration > 0) {
      requestPose(pingPongTime(elapsed * Number($('speed').value)));
    }
    if (headlight) { camera.getWorldDirection(headlightDirection); headlight.position.copy(camera.position); headlight.target.position.copy(camera.position).add(headlightDirection); headlight.target.updateMatrixWorld(); }
    controls.update(); if (dirty) { renderer.render(scene, camera); dirty = false; } animationId = requestAnimationFrame(animate);
  }
  animationId = requestAnimationFrame(animate);
  $('retry').onclick = start; start(); loadDataset();
} catch (error) { fail(`Интерактивный просмотр недоступен: ${error.message}`); loadDataset(); }
window.addEventListener('pagehide', () => { worker?.terminate(); clearTimeout(timer); cancelAnimationFrame(animationId); });
