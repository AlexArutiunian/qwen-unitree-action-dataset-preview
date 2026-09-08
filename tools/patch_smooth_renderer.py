#!/usr/bin/env python3
from pathlib import Path

# 1) Keep the exact MuJoCo mesh indexed instead of expanding every triangle
# corner with its imported normal. Geometry coordinates remain unchanged.
engine = Path('web/src/engine.worker.js')
s = engine.read_text()
start = s.index('    visible = [];\n    const geometries = [], transfers = [];')
end = s.index('    const motion = pendingMotion', start)
replacement = '''    visible = [];
    const geometries = [], transfers = [];
    let triangleCount = 0;
    for (let id = 0; id < model.ngeom; id++) {
      if (model.geom_group[id] !== 1) continue;
      if (model.geom_type[id] !== 7) throw new Error(`Unsupported visual geom type ${model.geom_type[id]}`);

      const mesh = model.geom_dataid[id];
      const va = model.mesh_vertadr[mesh], vn = model.mesh_vertnum[mesh];
      const fa = model.mesh_faceadr[mesh], fn = model.mesh_facenum[mesh];

      // Preserve the exact official compiled geometry as an indexed mesh.
      // Shading normals are generated in the renderer with a crease angle;
      // vertex positions and triangle topology are never changed.
      const vertices = new Float32Array(model.mesh_vert.subarray(va * 3, (va + vn) * 3));
      const faces = new Uint32Array(model.mesh_face.subarray(fa * 3, (fa + fn) * 3));
      const rgba = new Float32Array(model.geom_rgba.subarray(id * 4, id * 4 + 4));

      visible.push(id);
      triangleCount += fn;
      geometries.push({ vertices, faces, rgba, triangleCount: fn });
      transfers.push(vertices.buffer, faces.buffer, rgba.buffer);
    }
    for (const file of files) mj.FS.unlink(`/model/${file.replace(/\\.gz$/, '')}`);
    initializing = false;
    postMessage({ type: 'ready', geometries, triangles: triangleCount, format: mode, compileMs: performance.now() - start, nq: model.nq, njnt: model.njnt, version: message.version }, transfers);
'''
s = s[:start] + replacement + s[end:]
engine.write_text(s)

# 2) Three.js: exact indexed positions + crease-angle normals + neutral PBR IBL.
main = Path('web/src/main.js')
s = main.read_text()
old_imports = "import { OrbitControls } from 'three/addons/controls/OrbitControls.js';\n"
new_imports = old_imports + "import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';\nimport { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';\n"
if old_imports not in s:
    raise SystemExit('OrbitControls import anchor not found')
s = s.replace(old_imports, new_imports, 1)

old_state = "let renderer, scene, camera, controls, meshes = [], bounds, initialValues, software = false, animationId, dirty = true;"
new_state = "let renderer, scene, camera, controls, meshes = [], bounds, initialValues, environmentTexture, software = false, animationId, dirty = true;"
if old_state not in s:
    raise SystemExit('renderer state anchor not found')
s = s.replace(old_state, new_state, 1)

start = s.index('      meshes = data.geometries.map(g => {')
end = s.index('      const triangles = Number(data.triangles || 0)', start)
replacement = '''      meshes = data.geometries.map(g => {
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

        const color = new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]);
        const material = software
          ? new THREE.MeshLambertMaterial({ color })
          : new THREE.MeshPhysicalMaterial({
              color,
              roughness: .42,
              metalness: .08,
              clearcoat: .16,
              clearcoatRoughness: .48,
              envMapIntensity: .9,
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
'''
s = s[:start] + replacement + s[end:]

old_metrics = "      $('metrics').textContent = `${data.format.toUpperCase()} · HQ official mesh · ${triangles} граней · ${data.compileMs.toFixed(0)} мс · ${meshes.length} деталей · nq=${data.nq}`;"
new_metrics = "      $('metrics').textContent = `${data.format.toUpperCase()} · exact official mesh · ${triangles} граней · smooth crease 50° · ${data.compileMs.toFixed(0)} мс · ${meshes.length} деталей · nq=${data.nq}`;"
if old_metrics not in s:
    raise SystemExit('metrics anchor not found')
s = s.replace(old_metrics, new_metrics, 1)

old_append = "  $('viewport').append(renderer.domElement);\n"
new_append = """  $('viewport').append(renderer.domElement);
  if (!software) {
    const pmrem = new THREE.PMREMGenerator(renderer);
    environmentTexture = pmrem.fromScene(new RoomEnvironment(), .04).texture;
    scene.environment = environmentTexture;
    pmrem.dispose();
  }
"""
if old_append not in s:
    raise SystemExit('viewport append anchor not found')
s = s.replace(old_append, new_append, 1)

# Softer studio lighting: IBL now carries most surface illumination, reducing
# harsh per-triangle contrast while keeping shape readable.
s = s.replace(
    "scene.add(new THREE.HemisphereLight(0xcbdcff, 0x36465e, software ? .65 : 2));",
    "scene.add(new THREE.HemisphereLight(0xcbdcff, 0x36465e, software ? .65 : 1.05));",
    1
)
s = s.replace(
    "const light = new THREE.DirectionalLight(0xffffff, software ? .8 : 3);",
    "const light = new THREE.DirectionalLight(0xffffff, software ? .8 : 1.85);",
    1
)
s = s.replace(
    "const rim = new THREE.DirectionalLight(0x7ecde8, software ? .3 : 2);",
    "const rim = new THREE.DirectionalLight(0x7ecde8, software ? .3 : .75);",
    1
)

old_end = "window.addEventListener('pagehide', () => { worker?.terminate(); clearTimeout(timer); cancelAnimationFrame(animationId); });"
new_end = "window.addEventListener('pagehide', () => { worker?.terminate(); clearTimeout(timer); cancelAnimationFrame(animationId); environmentTexture?.dispose(); });"
if old_end not in s:
    raise SystemExit('pagehide anchor not found')
s = s.replace(old_end, new_end, 1)
main.write_text(s)

# 3) Explain the rendering model in the UI.
html = Path('web/index.html')
s = html.read_text()
old = 'Unitree G1 web-модель использует 29 суставов и исходные полноразмерные visual mesh из официального Unitree MJCF без decimation.'
new = 'Unitree G1 web-модель использует 29 суставов и исходные полноразмерные visual mesh из официального Unitree MJCF без decimation. Координаты и topology исходной сетки не меняются; гладкость поверхности создаётся только нормалями с crease-angle 50°, поэтому настоящие острые кромки сохраняются.'
if old not in s:
    raise SystemExit('HTML renderer description anchor not found')
html.write_text(s.replace(old, new, 1))
