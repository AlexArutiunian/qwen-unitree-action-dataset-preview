from pathlib import Path

p = Path('web/src/main.js')
s = p.read_text()

repls = {
    "const distance = radius / Math.sin(Math.min(halfV, halfH)) * 1.22;": "const distance = radius / Math.sin(Math.min(halfV, halfH)) * 1.12;",
    "scene.add(new THREE.AmbientLight(0xe8eef4, software ? .18 : .14));": "scene.add(new THREE.AmbientLight(0xe8eef4, software ? .18 : .17));",
    "scene.add(new THREE.HemisphereLight(0xc5d6ea, 0x31475a, software ? .45 : .28));": "scene.add(new THREE.HemisphereLight(0xc5d6ea, 0x31475a, software ? .45 : .34));",
    "const light = new THREE.DirectionalLight(0xfff6ec, software ? .85 : 2.05);": "const light = new THREE.DirectionalLight(0xfff6ec, software ? .85 : 2.30);",
    "headlight = new THREE.DirectionalLight(0xd4e4f0, software ? .25 : .72);": "headlight = new THREE.DirectionalLight(0xd4e4f0, software ? .25 : .84);",
}

for old, new in repls.items():
    if old not in s:
        raise SystemExit(f'anchor not found: {old}')
    s = s.replace(old, new, 1)

p.write_text(s)
