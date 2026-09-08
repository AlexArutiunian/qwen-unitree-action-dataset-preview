#!/usr/bin/env python3
"""One-shot source migration for the full-resolution G1 browser renderer."""
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f'{label}: anchor not found')
    if text.count(old) != 1:
        raise SystemExit(f'{label}: expected one anchor, found {text.count(old)}')
    return text.replace(old, new, 1)


engine = Path('web/src/engine.worker.js')
s = engine.read_text()
old_geometry = """    visible = []; const geometries = [];
    for (let id = 0; id < model.ngeom; id++) {
      if (model.geom_group[id] !== 1) continue;
      if (model.geom_type[id] !== 7) throw new Error(`Unsupported visual geom type ${model.geom_type[id]}`);
      const mesh = model.geom_dataid[id], va = model.mesh_vertadr[mesh], vn = model.mesh_vertnum[mesh];
      const fa = model.mesh_faceadr[mesh], fn = model.mesh_facenum[mesh];
      visible.push(id);
      geometries.push({ vertices: Array.from(model.mesh_vert.subarray(va * 3, (va + vn) * 3)), faces: Array.from(model.mesh_face.subarray(fa * 3, (fa + fn) * 3)), rgba: Array.from(model.geom_rgba.subarray(id * 4, id * 4 + 4)) });
    }
"""
new_geometry = """    visible = [];
    const geometries = [], transfers = [];
    let triangleCount = 0;
    for (let id = 0; id < model.ngeom; id++) {
      if (model.geom_group[id] !== 1) continue;
      if (model.geom_type[id] !== 7) throw new Error(`Unsupported visual geom type ${model.geom_type[id]}`);

      const mesh = model.geom_dataid[id];
      const va = model.mesh_vertadr[mesh], fa = model.mesh_faceadr[mesh], fn = model.mesh_facenum[mesh];
      const na = model.mesh_normaladr[mesh], nn = model.mesh_normalnum[mesh];
      const vertices = new Float32Array(fn * 9);
      const normals = new Float32Array(fn * 9);

      // Expand each triangle corner so MuJoCo's per-corner normal indices are
      // preserved exactly. This keeps CAD-like sharp edges while smoothing the
      // curved parts according to the official mesh compiler.
      for (let face = 0; face < fn; face++) {
        for (let corner = 0; corner < 3; corner++) {
          const faceSlot = (fa + face) * 3 + corner;
          const vi = model.mesh_face[faceSlot];
          const ni = model.mesh_facenormal[faceSlot];
          const dst = (face * 3 + corner) * 3;
          const srcV = (va + vi) * 3;
          vertices[dst] = model.mesh_vert[srcV];
          vertices[dst + 1] = model.mesh_vert[srcV + 1];
          vertices[dst + 2] = model.mesh_vert[srcV + 2];

          if (na >= 0 && nn > 0 && ni >= 0) {
            const srcN = (na + ni) * 3;
            normals[dst] = model.mesh_normal[srcN];
            normals[dst + 1] = model.mesh_normal[srcN + 1];
            normals[dst + 2] = model.mesh_normal[srcN + 2];
          }
        }
      }

      const rgba = new Float32Array(model.geom_rgba.subarray(id * 4, id * 4 + 4));
      visible.push(id);
      triangleCount += fn;
      geometries.push({ vertices, normals, rgba, triangleCount: fn });
      transfers.push(vertices.buffer, normals.buffer, rgba.buffer);
    }
"""
s = replace_once(s, old_geometry, new_geometry, 'engine geometry extraction')
old_ready = "    postMessage({ type: 'ready', geometries, format: mode, compileMs: performance.now() - start, nq: model.nq, njnt: model.njnt, version: message.version });"
new_ready = "    postMessage({ type: 'ready', geometries, triangles: triangleCount, format: mode, compileMs: performance.now() - start, nq: model.nq, njnt: model.njnt, version: message.version }, transfers);"
s = replace_once(s, old_ready, new_ready, 'engine ready payload')
engine.write_text(s)


main = Path('web/src/main.js')
s = main.read_text()
old_mesh = """      meshes = data.geometries.map(g => {
        const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.Float32BufferAttribute(g.vertices, 3)); geometry.setIndex(g.faces); geometry.computeVertexNormals();
        const material = software ? new THREE.MeshLambertMaterial({ color: new THREE.Color(...g.rgba.slice(0, 3)) }) : new THREE.MeshStandardMaterial({ color: new THREE.Color(...g.rgba.slice(0, 3)), roughness: .55, metalness: .22 });
        const mesh = new THREE.Mesh(geometry, material); mesh.matrixAutoUpdate = false; mesh.castShadow = true; mesh.receiveShadow = true; scene.add(mesh); return mesh;
      });
      $('metrics').textContent = `${data.format.toUpperCase()} · ${data.compileMs.toFixed(0)} мс · ${meshes.length} визуальных сеток · nq=${data.nq}`;
"""
new_mesh = """      meshes = data.geometries.map(g => {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(g.vertices, 3));
        if (g.normals && g.normals.length === g.vertices.length) {
          geometry.setAttribute('normal', new THREE.Float32BufferAttribute(g.normals, 3));
        } else {
          geometry.computeVertexNormals();
        }
        geometry.computeBoundingSphere();

        const color = new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]);
        const material = software
          ? new THREE.MeshLambertMaterial({ color })
          : new THREE.MeshStandardMaterial({
              color,
              roughness: .34,
              metalness: .16,
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
      $('metrics').textContent = `${data.format.toUpperCase()} · HQ official mesh · ${triangles} граней · ${data.compileMs.toFixed(0)} мс · ${meshes.length} деталей · nq=${data.nq}`;
"""
s = replace_once(s, old_mesh, new_mesh, 'main mesh creation')
old_renderer = "renderer.outputColorSpace = THREE.SRGBColorSpace; }"
new_renderer = "renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.08; }"
s = replace_once(s, old_renderer, new_renderer, 'renderer tone mapping')
main.write_text(s)


html = Path('web/index.html')
s = html.read_text()
s = replace_once(
    s,
    'Unitree G1 web-модель использует 29 суставов.',
    'Unitree G1 web-модель использует 29 суставов и исходные полноразмерные visual mesh из официального Unitree MJCF без decimation.',
    'model description',
)
html.write_text(s)

print('HQ renderer source migration applied')
