from pathlib import Path

main = Path('web/src/main.js')
s = main.read_text()

replacements = [
    ("import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';\n", ""),
    (
        "let renderer, scene, camera, controls, meshes = [], bounds, initialValues, environmentTexture, software = false, animationId, dirty = true;\nconst matrix = new THREE.Matrix4(), center = new THREE.Vector3(), direction = new THREE.Vector3(3, -2, 0.9);",
        "let renderer, scene, camera, controls, meshes = [], bounds, initialValues, software = false, animationId, dirty = true, headlight;\nconst matrix = new THREE.Matrix4(), center = new THREE.Vector3(), direction = new THREE.Vector3(3, -2, 0.9), headlightDirection = new THREE.Vector3();"
    ),
    (
        "        // MuJoCo visual rgba values are authored as display colors. Convert\n"
        "        // from sRGB into Three.js' linear working space so 0.2 stays dark\n"
        "        // instead of looking like a washed-out mid gray.\n"
        "        const color = new THREE.Color().setRGB(g.rgba[0], g.rgba[1], g.rgba[2], THREE.SRGBColorSpace);\n"
        "        const material = software\n"
        "          ? new THREE.MeshLambertMaterial({ color })\n"
        "          : new THREE.MeshPhysicalMaterial({\n"
        "              color,\n"
        "              roughness: .5,\n"
        "              metalness: .05,\n"
        "              clearcoat: .08,\n"
        "              clearcoatRoughness: .55,\n"
        "              envMapIntensity: .45,\n"
        "              flatShading: false,\n"
        "              transparent: g.rgba[3] < .999,\n"
        "              opacity: g.rgba[3]\n"
        "            });",
        "        // WBC-style material mapping. Keep MuJoCo's authored two-tone\n"
        "        // palette, but bias it slightly darker/cooler so the 0.7 shell reads\n"
        "        // as technical silver-gray instead of white plastic.\n"
        "        const luma = 0.2126 * g.rgba[0] + 0.7152 * g.rgba[1] + 0.0722 * g.rgba[2];\n"
        "        const grayScale = luma > .35 ? .88 : .80;\n"
        "        const color = new THREE.Color().setRGB(\n"
        "          Math.min(1, g.rgba[0] * grayScale),\n"
        "          Math.min(1, g.rgba[1] * grayScale),\n"
        "          Math.min(1, g.rgba[2] * grayScale),\n"
        "          THREE.SRGBColorSpace\n"
        "        );\n"
        "        const shininess = luma > .35 ? .68 : .48;\n"
        "        const material = software\n"
        "          ? new THREE.MeshLambertMaterial({ color })\n"
        "          : new THREE.MeshPhysicalMaterial({\n"
        "              color,\n"
        "              roughness: 1 - shininess,\n"
        "              metalness: 0,\n"
        "              specularIntensity: .46,\n"
        "              specularColor: new THREE.Color(0xd4e4f0),\n"
        "              flatShading: false,\n"
        "              transparent: g.rgba[3] < .999,\n"
        "              opacity: g.rgba[3]\n"
        "            });"
    ),
    (
        "$('metrics').textContent = `${data.format.toUpperCase()} · exact official mesh · ${triangles} граней · smooth crease 50° · ${data.compileMs.toFixed(0)} мс · ${meshes.length} деталей · nq=${data.nq}`;",
        "$('metrics').textContent = `${data.format.toUpperCase()} · exact official mesh · ${triangles} граней · smooth crease 50° · WBC gray · ${data.compileMs.toFixed(0)} мс · ${meshes.length} деталей · nq=${data.nq}`;"
    ),
    ("scene = new THREE.Scene(); scene.fog = new THREE.Fog('#121a24', 9, 22);", "scene = new THREE.Scene(); scene.fog = new THREE.Fog('#16283a', 20, 60);"),
    (
        "  if (context) { renderer = new THREE.WebGLRenderer({ canvas, context, antialias: true, alpha: true }); renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap; renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = .86; }",
        "  if (context) { renderer = new THREE.WebGLRenderer({ canvas, context, antialias: true, alpha: true }); renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap; renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.NoToneMapping; }"
    ),
    (
        "  if (!software) {\n"
        "    const pmrem = new THREE.PMREMGenerator(renderer);\n"
        "    environmentTexture = pmrem.fromScene(new RoomEnvironment(), .04).texture;\n"
        "    scene.environment = environmentTexture;\n"
        "    pmrem.dispose();\n"
        "  }\n",
        ""
    ),
    (
        "  scene.add(new THREE.HemisphereLight(0xcbdcff, 0x36465e, software ? .65 : .82));\n"
        "  const light = new THREE.DirectionalLight(0xffffff, software ? .8 : 1.35); light.position.set(3, -4, 6); light.castShadow = true; light.shadow.mapSize.set(2048, 2048); light.shadow.camera.left = -3; light.shadow.camera.right = 3; light.shadow.camera.top = 3; light.shadow.camera.bottom = -3; scene.add(light);\n"
        "  const rim = new THREE.DirectionalLight(0x7ecde8, software ? .3 : .42); rim.position.set(-3, 3, 4); scene.add(rim);\n"
        "  const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: 0x192331, roughness: .9 })); floor.position.z = -.007; floor.receiveShadow = true; if (!software) scene.add(floor);\n"
        "  const grid = new THREE.GridHelper(12, 60, 0x43586d, 0x263647); grid.rotation.x = Math.PI / 2; grid.position.z = -.005; if (software) { grid.material.vertexColors = false; grid.material.color.set(0x314154); } scene.add(grid);",
        "  // Lighting follows the WBC viewer: low ambient, an overhead MuJoCo-like\n"
        "  // key, cool floor bounce and a soft camera headlight. No HDR environment\n"
        "  // and no filmic tone mapping, so gray body panels keep their separation.\n"
        "  scene.add(new THREE.AmbientLight(0xe8eef4, software ? .18 : .14));\n"
        "  scene.add(new THREE.HemisphereLight(0xc5d6ea, 0x31475a, software ? .45 : .28));\n"
        "  const light = new THREE.DirectionalLight(0xfff6ec, software ? .85 : 2.05);\n"
        "  light.position.set(1.2, -.5, 5.6); light.target.position.set(0, 0, .9); scene.add(light.target);\n"
        "  light.castShadow = true; light.shadow.mapSize.set(2048, 2048); light.shadow.camera.left = -3; light.shadow.camera.right = 3; light.shadow.camera.top = 3; light.shadow.camera.bottom = -3; light.shadow.bias = -.0002; light.shadow.normalBias = .02; scene.add(light);\n"
        "  headlight = new THREE.DirectionalLight(0xd4e4f0, software ? .25 : .72); headlight.castShadow = false; scene.add(headlight.target); scene.add(headlight);\n"
        "  const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: 0x2a455c, roughness: .96, metalness: 0 })); floor.position.z = -.007; floor.receiveShadow = true; if (!software) scene.add(floor);\n"
        "  const grid = new THREE.GridHelper(12, 60, 0x429eb0, 0x2f7a8a); grid.rotation.x = Math.PI / 2; grid.position.z = -.005; if (software) { grid.material.vertexColors = false; grid.material.color.set(0x395668); } scene.add(grid);"
    ),
    (
        "    controls.update(); if (dirty) { renderer.render(scene, camera); dirty = false; } animationId = requestAnimationFrame(animate);",
        "    if (headlight) { camera.getWorldDirection(headlightDirection); headlight.position.copy(camera.position); headlight.target.position.copy(camera.position).add(headlightDirection); headlight.target.updateMatrixWorld(); }\n    controls.update(); if (dirty) { renderer.render(scene, camera); dirty = false; } animationId = requestAnimationFrame(animate);"
    ),
    (
        "window.addEventListener('pagehide', () => { worker?.terminate(); clearTimeout(timer); cancelAnimationFrame(animationId); environmentTexture?.dispose(); });",
        "window.addEventListener('pagehide', () => { worker?.terminate(); clearTimeout(timer); cancelAnimationFrame(animationId); });"
    ),
]

for old, new in replacements:
    if old not in s:
        raise SystemExit(f'main.js anchor not found: {old[:120]!r}')
    s = s.replace(old, new, 1)
main.write_text(s)

css_path = Path('web/src/style.css')
css = css_path.read_text()
css_replacements = [
    ("#viewport{position:absolute;inset:0 0 98px;touch-action:none}", "#viewport{position:absolute;inset:132px 0 98px;touch-action:none}"),
    (".stage-title{position:absolute;top:28px;left:28px;pointer-events:none;z-index:2;max-width:min(60%,760px)}", ".stage-title{position:absolute;top:0;left:0;right:0;height:132px;padding:20px 106px 14px 28px;pointer-events:none;z-index:4;max-width:none;background:linear-gradient(180deg,#111923f2 0%,#111923dc 72%,#11192300 100%);border-bottom:1px solid #ffffff0a}"),
    ("h1{font-size:clamp(20px,2vw,32px);line-height:1.22;letter-spacing:-.03em;margin:12px 0;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}", "h1{font-size:clamp(22px,2vw,32px);line-height:1.18;letter-spacing:-.03em;margin:9px 0 7px;max-width:min(900px,78%);display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}"),
    (".stage-title p{color:#a0afc0;font-size:14px}", ".stage-title p{color:#a0afc0;font-size:14px;margin:0}"),
    (".stage-title{top:18px;left:20px;max-width:calc(100% - 112px)}", ".stage-title{top:0;left:0;right:0;height:124px;padding:16px 92px 12px 20px;max-width:none}#viewport{top:124px}"),
    (".stage-title{top:14px;left:14px;max-width:calc(100% - 86px)}", ".stage-title{top:0;left:0;right:0;height:108px;padding:12px 76px 10px 14px;max-width:none}"),
    (".stage-title h1{font-size:20px;line-height:1.18;margin:8px 0;-webkit-line-clamp:2}", ".stage-title h1{font-size:19px;line-height:1.16;margin:6px 0 5px;max-width:100%;-webkit-line-clamp:2}"),
    ("#viewport{bottom:108px}", "#viewport{top:108px;bottom:108px}"),
    (".stage-title{max-width:calc(100% - 78px)}", ".stage-title{height:104px;padding-right:68px}#viewport{top:104px}"),
]
for old, new in css_replacements:
    if old not in css:
        raise SystemExit(f'style.css anchor not found: {old[:120]!r}')
    css = css.replace(old, new, 1)
css_path.write_text(css)
