#!/usr/bin/env python3
"""Build browser assets from the pinned official Unitree G1 29-DoF model.

By default the visual STL files are copied byte-for-byte from the official
Unitree source. Set --faces-per-mesh to a positive value to build a reduced LOD,
or --hull for the CPU-only convex fallback.

Requires: mujoco==3.12.0 trimesh==5.1.0 fast-simplification==0.2.0 numpy==2.3.5.
Source: unitreerobotics/unitree_mujoco @ 1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import trimesh

PINNED_REVISION = '1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path, help='unitree_mujoco checkout root')
parser.add_argument('--output', type=Path, default=Path('web/public/model'))
parser.add_argument('--hull', action='store_true', help='Convex visual shells for CPU-only 3D fallback')
parser.add_argument(
    '--faces-per-mesh',
    type=int,
    default=0,
    help='Maximum faces per mesh; 0 preserves the official STL geometry exactly',
)
args = parser.parse_args()

if mujoco.__version__ != '3.12.0':
    raise SystemExit('MuJoCo 3.12.0 is required to match the browser MJB ABI')
if args.faces_per_mesh < 0 or 0 < args.faces_per_mesh < 100:
    raise SystemExit('faces-per-mesh must be 0 (exact) or at least 100')

revision = subprocess.check_output(
    ['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True
).strip()
if revision != PINNED_REVISION:
    raise SystemExit('Use the pinned official Unitree source revision for reproducible assets')

source = args.source / 'unitree_robots/g1'
root = ET.parse(source / 'g1_29dof.xml')
output = args.output
(output / 'meshes').mkdir(parents=True, exist_ok=True)

# Remove stale mesh files so switching between LODs cannot leave orphan assets.
for stale in (output / 'meshes').glob('*'):
    if stale.is_file():
        stale.unlink()

files = ['g1.xml']
original_faces = output_faces = 0
exact_visual_meshes = not args.hull and args.faces_per_mesh == 0

for element in root.findall('./asset/mesh'):
    path = source / 'meshes' / element.attrib['file']
    probe = trimesh.load(path, force='mesh', process=False)
    original_faces += len(probe.faces)
    filename = path.stem + '.stl'
    destination = output / 'meshes' / filename

    if exact_visual_meshes:
        # Preserve the authoritative Unitree triangulation byte-for-byte.
        shutil.copyfile(path, destination)
        output_faces += len(probe.faces)
    else:
        mesh = trimesh.load(path, force='mesh', process=True)
        if args.hull:
            mesh = mesh.convex_hull
        if (
            args.faces_per_mesh > 0
            and len(mesh.faces) > args.faces_per_mesh
            and not (args.hull and path.stem in {'head_link', 'torso_link', 'pelvis'})
        ):
            mesh = mesh.simplify_quadric_decimation(
                face_count=args.faces_per_mesh, aggression=9
            )
        output_faces += len(mesh.faces)
        mesh.export(destination, file_type='stl')

    element.set('file', filename)
    files.append('meshes/' + filename)

root.write(output / 'g1.xml', encoding='utf-8', xml_declaration=True)
shutil.copyfile(args.source / 'LICENSE', output / 'LICENSE')

model = mujoco.MjModel.from_xml_path(str(output / 'g1.xml'))
mujoco.mj_saveModel(model, str(output / 'g1.mjb'))
(output / 'g1.mjb.gz').write_bytes(
    gzip.compress((output / 'g1.mjb').read_bytes(), mtime=0)
)

asset_files = files + ['g1.mjb', 'g1.mjb.gz']
manifest = {
    'source': 'https://github.com/unitreerobotics/unitree_mujoco',
    'revision': PINNED_REVISION,
    'source_xml_sha256': hashlib.sha256((source / 'g1_29dof.xml').read_bytes()).hexdigest(),
    'mujoco': mujoco.__version__,
    'visual_hulls': args.hull,
    'exact_visual_meshes': exact_visual_meshes,
    'faces_per_mesh': args.faces_per_mesh,
    'model': 'g1_29dof',
    'binary': 'g1.mjb.gz',
    'files': files,
    'original_faces': original_faces,
    'web_faces': output_faces,
    'bytes': {p: (output / p).stat().st_size for p in asset_files},
    'sha256': {
        p: hashlib.sha256((output / p).read_bytes()).hexdigest()
        for p in asset_files
    },
}
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

print(json.dumps({
    'original_faces': original_faces,
    'web_faces': output_faces,
    'exact_visual_meshes': exact_visual_meshes,
}))
print(
    'MJCF + meshes:',
    sum(manifest['bytes'][p] for p in files),
    'bytes; MJB:',
    manifest['bytes']['g1.mjb'],
)
