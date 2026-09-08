#!/usr/bin/env python3
"""Build web LOD from official Unitree g1_29dof.xml without altering kinematics.
Requires: mujoco==3.12.0 trimesh==5.1.0 fast-simplification==0.2.0 numpy==2.3.5.
Source: unitreerobotics/unitree_mujoco @ 1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d.
"""
import argparse
import hashlib
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import mujoco
import trimesh

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path, help='unitree_mujoco checkout root')
parser.add_argument('--output', type=Path, default=Path('web/public/model'))
parser.add_argument('--hull', action='store_true', help='Convex visual shells for CPU-only 3D fallback')
parser.add_argument('--faces-per-mesh', type=int, default=1800)
args = parser.parse_args()
if mujoco.__version__ != '3.12.0':
    raise SystemExit('MuJoCo 3.12.0 is required to match the browser MJB ABI')
if args.faces_per_mesh < 100:
    raise SystemExit('faces-per-mesh must be at least 100')
revision = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
if revision != '1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d':
    raise SystemExit('Use the pinned official Unitree source revision for reproducible assets')
source = args.source / 'unitree_robots/g1'
root = ET.parse(source / 'g1_29dof.xml')
output = args.output
(output / 'meshes').mkdir(parents=True, exist_ok=True)
files = ['g1.xml']
original_faces = reduced_faces = 0
for element in root.findall('./asset/mesh'):
    path = source / 'meshes' / element.attrib['file']
    mesh = trimesh.load(path, force='mesh', process=True)
    original_faces += len(mesh.faces)
    if args.hull:
        mesh = mesh.convex_hull
    if len(mesh.faces) > args.faces_per_mesh and not (args.hull and path.stem in {'head_link','torso_link','pelvis'}):
        mesh = mesh.simplify_quadric_decimation(face_count=args.faces_per_mesh, aggression=9)
    reduced_faces += len(mesh.faces)
    filename = path.stem + '.stl'
    # Binary STL: portable, compact, reproducible; original mesh coordinate frame.
    mesh.export(output / 'meshes' / filename, file_type='stl')
    element.set('file', filename)
    files.append('meshes/' + filename)
root.write(output / 'g1.xml', encoding='utf-8', xml_declaration=True)
shutil.copyfile(args.source / 'LICENSE', output / 'LICENSE')
model = mujoco.MjModel.from_xml_path(str(output / 'g1.xml'))
mujoco.mj_saveModel(model, str(output / 'g1.mjb'))
(output / 'g1.mjb.gz').write_bytes(gzip.compress((output / 'g1.mjb').read_bytes(), mtime=0))
manifest = {
    'source': 'https://github.com/unitreerobotics/unitree_mujoco',
    'revision': '1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d',
    'source_xml_sha256': hashlib.sha256((source/'g1_29dof.xml').read_bytes()).hexdigest(),
    'mujoco': mujoco.__version__, 'visual_hulls': args.hull, 'model': 'g1_29dof',
    'binary': 'g1.mjb.gz', 'files': files, 'original_faces': original_faces, 'web_faces': reduced_faces,
    'bytes': {p: (output/p).stat().st_size for p in files + ['g1.mjb', 'g1.mjb.gz']},
    'sha256': {p: hashlib.sha256((output/p).read_bytes()).hexdigest() for p in files + ['g1.mjb', 'g1.mjb.gz']},
}
(output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ['original_faces','web_faces']}))
print('MJCF + meshes:', sum(manifest['bytes'][p] for p in files), 'bytes; MJB:',manifest['bytes']['g1.mjb'])
