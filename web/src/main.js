import * as THREE from 'three';
import { SVGRenderer } from 'three/addons/renderers/SVGRenderer.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { sample91 } from './sample.js';
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
let renderer, scene, camera, controls, meshes = [], bounds, initialValues, software = false, animationId, dirty = true;
const matrix = new THREE.Matrix4(), center = new THREE.Vector3(), direction = new THREE.Vector3(3, -2, 0.9);

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
  if (view === 'front') direction.set(1, 0, 0.08);
  if (view === 'side') direction.set(0, -1, 0.08);
  if (view === 'back') direction.set(-1, 0, 0.08);
  if (view === 'fit') direction.copy(camera.position).sub(controls.target);
  const size = bounds.getSize(new THREE.Vector3()); bounds.getCenter(center);
  const radius = Math.max(0.25, size.length() / 2);
  const halfV = THREE.MathUtils.degToRad(camera.fov / 2), halfH = Math.atan(Math.tan(halfV) * camera.aspect);
  const distance = radius / Math.sin(Math.min(halfV, halfH)) * 1.22;
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
    $('datasetStatus').textContent = `Загружено ${actions.length} actions · train + validation + reserved`;
    populateActions();
    const requested = Number(new URLSearchParams(location.search).get('action')) || 91;
    const action = unique.get(requested) || unique.get(91) || actions[0];
    if (action) { selectAction(action, false); $('actionSelect').value = String(action.sample_id); }
  } catch (error) {
    $('datasetStatus').textContent = `Полный список недоступен: ${error.message}. Оставлен sample 91.`;
    $('actionSelect').disabled = true; $('actionSearch').disabled = true; $('splitFilter').disabled = true;
  }
}

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
$('prevAction').onclick = () => {
  const index = actions.findIndex(action => action.sample_id === currentAction.sample_id); if (index > 0) { selectAction(actions[index - 1]); populateActions(); }
};
$('nextAction').onclick = () => {
  const index = actions.findIndex(action => action.sample_id === currentAction.sample_id); if (index >= 0 && index < actions.length - 1) { selectAction(actions[index + 1]); populateActions(); }
};
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
        const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.Float32BufferAttribute(g.vertices, 3)); geometry.setIndex(g.faces); geometry.computeVertexNormals();
        const material = software ? new THREE.MeshLambertMaterial({ color: new THREE.Color(...g.rgba.slice(0, 3)) }) : new THREE.MeshStandardMaterial({ color: new THREE.Color(...g.rgba.slice(0, 3)), roughness: .55, metalness: .22 });
        const mesh = new THREE.Mesh(geometry, material); mesh.matrixAutoUpdate = false; mesh.castShadow = true; mesh.receiveShadow = true; scene.add(mesh); return mesh;
      });
      $('metrics').textContent = `${data.format.toUpperCase()} · ${data.compileMs.toFixed(0)} мс · ${meshes.length} визуальных сеток · nq=${data.nq}`;
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
  scene = new THREE.Scene(); scene.fog = new THREE.Fog('#121a24', 9, 22);
  camera = new THREE.PerspectiveCamera(36, 1, .01, 100); camera.up.set(0, 0, 1);
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('webgl2', { antialias: true, alpha: true });
  if (context) { renderer = new THREE.WebGLRenderer({ canvas, context, antialias: true, alpha: true }); renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap; renderer.outputColorSpace = THREE.SRGBColorSpace; }
  else { software = true; renderer = new SVGRenderer(); renderer.setQuality('low'); renderer.setClearColor(0x17212e); renderer.domElement.style.width = '100%'; renderer.domElement.style.height = '100%'; }
  $('viewport').append(renderer.domElement);
  renderer.domElement.addEventListener('webglcontextlost', e => { e.preventDefault(); fail('WebGL-контекст потерян. Обновите страницу.'); });
  controls = new OrbitControls(camera, renderer.domElement); controls.addEventListener('change', () => { dirty = true; }); controls.enableDamping = true; controls.dampingFactor = .08;
  scene.add(new THREE.HemisphereLight(0xcbdcff, 0x36465e, software ? .65 : 2));
  const light = new THREE.DirectionalLight(0xffffff, software ? .8 : 3); light.position.set(3, -4, 6); light.castShadow = true; light.shadow.mapSize.set(2048, 2048); light.shadow.camera.left = -3; light.shadow.camera.right = 3; light.shadow.camera.top = 3; light.shadow.camera.bottom = -3; scene.add(light);
  const rim = new THREE.DirectionalLight(0x7ecde8, software ? .3 : 2); rim.position.set(-3, 3, 4); scene.add(rim);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: 0x192331, roughness: .9 })); floor.position.z = -.007; floor.receiveShadow = true; if (!software) scene.add(floor);
  const grid = new THREE.GridHelper(12, 60, 0x43586d, 0x263647); grid.rotation.x = Math.PI / 2; grid.position.z = -.005; if (software) { grid.material.vertexColors = false; grid.material.color.set(0x314154); } scene.add(grid);
  new ResizeObserver(() => {
    const { width, height } = $('viewport').getBoundingClientRect(); renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix(); fit('fit');
  }).observe($('viewport'));
  let last = performance.now();
  function animate(now) {
    const elapsed = Math.max(0, (now - last) / 1000); last = now;
    if (playing && ready && duration > 0) {
      requestPose(pingPongTime(elapsed * Number($('speed').value)));
    }
    controls.update(); if (dirty) { renderer.render(scene, camera); dirty = false; } animationId = requestAnimationFrame(animate);
  }
  animationId = requestAnimationFrame(animate);
  $('retry').onclick = start; start(); loadDataset();
} catch (error) { fail(`Интерактивный просмотр недоступен: ${error.message}`); loadDataset(); }
window.addEventListener('pagehide', () => { worker?.terminate(); clearTimeout(timer); cancelAnimationFrame(animationId); });
