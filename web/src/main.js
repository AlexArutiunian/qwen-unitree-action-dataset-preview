import * as THREE from 'three';
import { SVGRenderer } from 'three/addons/renderers/SVGRenderer.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { sample91 } from './sample.js';
import './style.css';
const $=id=>document.getElementById(id);
$('command').textContent=sample91.text;
$('json').textContent=JSON.stringify(sample91.program,null,2);
const labels=['Л. плечо: pitch','Л. плечо: roll','Л. плечо: yaw','Л. локоть','П. плечо: pitch','П. плечо: roll','П. плечо: yaw','П. локоть'];
const targets=sample91.program[0].frame;
const cells=targets.map((joint,i)=>{
 const row=document.createElement('tr');
 const name=document.createElement('td');name.textContent=labels[i];name.title=joint.name;
 const actual=document.createElement('td');actual.textContent='—';actual.dataset.joint=joint.name;
 const goal=document.createElement('td');goal.textContent=joint.angle.toFixed(1);
 row.append(name,actual,goal);$('joints').append(row);return actual;
});
let worker, timer, playing=false, time=0, duration=1.5, ready=false, busy=false, wanted=0;
let renderer,scene,camera,controls,meshes=[],bounds,initialValues,software=false,animationId,dirty=true;
const matrix=new THREE.Matrix4(), center=new THREE.Vector3(), direction=new THREE.Vector3(3,-2,0.9);
function fail(message) {
 ready=false;playing=false;clearTimeout(timer);worker?.terminate();
 $('fallback').hidden=false;$('loading').textContent=message;$('retry').hidden=false;$('retry').onclick=()=>location.reload();
 $('status').textContent='3D недоступен';$('play').disabled=true;$('reset').disabled=true;$('seek').disabled=true;
}
function clock() {$('clock').textContent=`${time.toFixed(2)} / ${duration.toFixed(2)} с`;$('seek').value=time;}
function requestPose(value) {
 time=Math.max(0,Math.min(duration,value)); wanted=time;clock();
 if(!ready||busy)return;busy=true;worker.postMessage({type:'seek',time:wanted});
}
function fit(view) {
 if(!bounds)return;
 if(view==='front')direction.set(1,0,0.08);
 if(view==='side')direction.set(0,-1,0.08);
 if(view==='back')direction.set(-1,0,0.08);
 if(view==='fit')direction.copy(camera.position).sub(controls.target);
 const size=bounds.getSize(new THREE.Vector3());bounds.getCenter(center);
 const radius=size.length()/2;
 const halfV=THREE.MathUtils.degToRad(camera.fov/2),halfH=Math.atan(Math.tan(halfV)*camera.aspect);
 const distance=radius/Math.sin(Math.min(halfV,halfH))*1.22;
 camera.position.copy(center).addScaledVector(direction.normalize(),distance);
 camera.near=.01;camera.far=Math.max(100,distance*10);camera.updateProjectionMatrix();
 controls.target.copy(center);controls.minDistance=radius*.5;controls.maxDistance=distance*5;controls.update();
}
function toggle() {if(!ready)return;if(time>=duration)requestPose(0);playing=!playing;$('play').textContent=playing?'Пауза':'Воспроизвести';}
function reset() {playing=false;$('play').textContent='Воспроизвести';requestPose(0);}
$('play').onclick=toggle;$('reset').onclick=reset;
$('seek').oninput=event=>{playing=false;$('play').textContent='Воспроизвести';requestPose(Number(event.target.value));};
for(const button of document.querySelectorAll('[data-view]'))button.onclick=()=>fit(button.dataset.view);
document.addEventListener('keydown',event=>{
 if(/INPUT|SELECT|TEXTAREA|BUTTON|SUMMARY/.test(event.target.tagName)||event.target.isContentEditable)return;
 if(event.code==='Space'){event.preventDefault();toggle();}
 if(event.code==='KeyR')reset();
 if(event.code==='ArrowLeft'||event.code==='ArrowRight'){event.preventDefault();playing=false;requestPose(time+(event.code==='ArrowLeft'?-.1:.1));$('play').textContent='Воспроизвести';}
});
function start() {
 worker?.terminate();clearTimeout(timer);ready=false;busy=false;initialValues=null;reset();
 $('retry').hidden=true;$('fallback').hidden=false;
 for(const mesh of meshes){scene.remove(mesh);mesh.geometry.dispose();mesh.material.dispose();}meshes=[];
 worker=new Worker(new URL('./engine.worker.js',import.meta.url),{type:'module'});
 timer=setTimeout(()=>fail('Загрузка заняла больше 60 секунд. Проверьте соединение и повторите попытку.'),60000);
 worker.onerror=event=>fail(event.message||'Ошибка запуска MuJoCo');
 worker.onmessage=({data})=>{
  if(data.type==='status'){$('status').textContent=data.text;$('loading').textContent=data.text;}
  if(data.type==='error')fail(data.text);
  if(data.type==='ready') {
   clearTimeout(timer);duration=data.duration;$('seek').max=duration;
   meshes=data.geometries.map(g=>{
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(g.vertices,3));geometry.setIndex(g.faces);geometry.computeVertexNormals();
    const material=software ? new THREE.MeshLambertMaterial({color:new THREE.Color(...g.rgba.slice(0,3))}) : new THREE.MeshStandardMaterial({color:new THREE.Color(...g.rgba.slice(0,3)),roughness:.55,metalness:.22});
    const mesh=new THREE.Mesh(geometry,material);mesh.matrixAutoUpdate=false;mesh.castShadow=true;mesh.receiveShadow=true;scene.add(mesh);return mesh;
   });
   bounds=new THREE.Box3(new THREE.Vector3(...data.bounds.min),new THREE.Vector3(...data.bounds.max));
   fit('front');ready=true;
   $('metrics').textContent=`${data.format.toUpperCase()} · ${data.compileMs.toFixed(0)} мс · ${meshes.length} визуальных сеток · nq=${data.nq}`;
   $('status').textContent=software?'G1 · Программный 3D':'G1 · WebGL';$('play').disabled=false;$('reset').disabled=false;$('seek').disabled=false;
  }
  if(data.type==='pose') {
   dirty=true;
   const t=data.transforms;
   for(let i=0;i<meshes.length;i++) {
    const p=i*12;
    matrix.set(t[p+3],t[p+4],t[p+5],t[p],t[p+6],t[p+7],t[p+8],t[p+1],t[p+9],t[p+10],t[p+11],t[p+2],0,0,0,1);
    meshes[i].matrix.copy(matrix);meshes[i].matrixWorldNeedsUpdate=true;
   }
   if(!initialValues)initialValues=data.values;
   targets.forEach((j,i)=>cells[i].textContent=data.values[j.name].toFixed(1));
   const targetNames=new Set(targets.map(j=>j.name));
   const neutral=Object.entries(data.values).every(([name,value])=>targetNames.has(name)||Math.abs(value-initialValues[name])<1e-8);
   const error=Math.max(...targets.map(j=>Math.abs(data.values[j.name]-j.angle)));
   $('check').textContent=data.time>=duration?`Цель: ошибка ${error.toFixed(6)}° · Остальные суставы ${neutral?'не изменены':'ИЗМЕНЕНЫ'}`:'Плавная интерполяция · Остальные суставы в qpos0';
   $('fallback').hidden=true;busy=false;
   if(Math.abs(wanted-data.time)>1e-6)requestPose(wanted);
  }
 };
 worker.postMessage({type:'init',base:new URL('./',document.baseURI).href,format:new URLSearchParams(location.search).get('format'),lite:software});
}
try {
 scene=new THREE.Scene();scene.fog=new THREE.Fog('#121a24',9,22);
 camera=new THREE.PerspectiveCamera(36,1,.01,100);camera.up.set(0,0,1);
 const canvas=document.createElement('canvas');
 const context=canvas.getContext('webgl2',{antialias:true,alpha:true});
 if(context){renderer=new THREE.WebGLRenderer({canvas,context,antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.outputColorSpace=THREE.SRGBColorSpace;}
 else{software=true;renderer=new SVGRenderer();renderer.setQuality('low');renderer.setClearColor(0x17212e);renderer.domElement.style.width='100%';renderer.domElement.style.height='100%';} 
 $('viewport').append(renderer.domElement);
 renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();fail('WebGL-контекст потерян. Обновите страницу.');});
 controls=new OrbitControls(camera,renderer.domElement);controls.addEventListener('change',()=>{dirty=true;});controls.enableDamping=true;controls.dampingFactor=.08;
 scene.add(new THREE.HemisphereLight(0xcbdcff,0x36465e,software?.65:2));
 const light=new THREE.DirectionalLight(0xffffff,software?.8:3);light.position.set(3,-4,6);light.castShadow=true;light.shadow.mapSize.set(2048,2048);light.shadow.camera.left=-3;light.shadow.camera.right=3;light.shadow.camera.top=3;light.shadow.camera.bottom=-3;scene.add(light);
 const rim=new THREE.DirectionalLight(0x7ecde8,software?.3:2);rim.position.set(-3,3,4);scene.add(rim);
 const floor=new THREE.Mesh(new THREE.PlaneGeometry(200,200),new THREE.MeshStandardMaterial({color:0x192331,roughness:.9}));floor.position.z=-.007;floor.receiveShadow=true;if(!software)scene.add(floor);
 const grid=new THREE.GridHelper(12,60,0x43586d,0x263647);grid.rotation.x=Math.PI/2;grid.position.z=-.005;if(software){grid.material.vertexColors=false;grid.material.color.set(0x314154);}scene.add(grid);
 new ResizeObserver(()=>{
  const {width,height}=$('viewport').getBoundingClientRect();renderer.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix();fit('fit');
 }).observe($('viewport'));
 let last=performance.now();
 function animate(now){
  const elapsed=Math.max(0,(now-last)/1000);last=now;
  if(playing&&ready){requestPose(time+elapsed*Number($('speed').value));if(time>=duration){playing=false;$('play').textContent='Воспроизвести';}}
  controls.update();if(dirty){renderer.render(scene,camera);dirty=false;}animationId=requestAnimationFrame(animate);
 }
 animationId=requestAnimationFrame(animate);
 $('retry').onclick=start;start();
} catch(error) {fail(`Интерактивный просмотр недоступен: ${error.message}`);}
window.addEventListener('pagehide',()=>{worker?.terminate();clearTimeout(timer);cancelAnimationFrame(animationId);});
