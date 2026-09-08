from pathlib import Path

p = Path('web/src/main.js')
s = p.read_text()

repls = {
"        const color = new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]);": "        // MuJoCo visual rgba values are authored as display colors. Convert\n        // from sRGB into Three.js' linear working space so 0.2 stays dark\n        // instead of looking like a washed-out mid gray.\n        const color = new THREE.Color().setRGB(g.rgba[0], g.rgba[1], g.rgba[2], THREE.SRGBColorSpace);",
"              roughness: .42,\n              metalness: .08,\n              clearcoat: .16,\n              clearcoatRoughness: .48,\n              envMapIntensity: .9,": "              roughness: .5,\n              metalness: .05,\n              clearcoat: .08,\n              clearcoatRoughness: .55,\n              envMapIntensity: .45,",
"renderer.toneMappingExposure = 1.08;": "renderer.toneMappingExposure = .86;",
"scene.add(new THREE.HemisphereLight(0xcbdcff, 0x36465e, software ? .65 : 2));": "scene.add(new THREE.HemisphereLight(0xcbdcff, 0x36465e, software ? .65 : .9));",
"const light = new THREE.DirectionalLight(0xffffff, software ? .8 : 3);": "const light = new THREE.DirectionalLight(0xffffff, software ? .8 : 1.55);",
"const rim = new THREE.DirectionalLight(0x7ecde8, software ? .3 : 2);": "const rim = new THREE.DirectionalLight(0x7ecde8, software ? .3 : .55);",
}

for old, new in repls.items():
    if old not in s:
        raise SystemExit(f'anchor not found: {old[:80]}')
    s = s.replace(old, new, 1)

p.write_text(s)
