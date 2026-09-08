import loadMujoco from '@mujoco/mujoco';
import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
import { compile, poseAt } from './timeline.js';
import { sample91 } from './sample.js';
let mj, model, data, timeline, joints, visible;
const status = text => postMessage({ type: 'status', text });
async function fetchBytes(url) {
 const response = await fetch(url, { signal: AbortSignal.timeout(30000) });
 if (!response.ok) throw new Error(`${response.status}: ${url}`);
 return new Uint8Array(await response.arrayBuffer());
}
function sendPose(time) {
 data.qpos.set(model.qpos0);
 const pose=poseAt(timeline,time);
 for (const joint of joints) data.qpos[joint.address]=pose[joint.name];
 mj.mj_forward(model,data);
 const transforms=new Float64Array(visible.length*12);
 for(let i=0;i<visible.length;i++) {
  const id=visible[i]; transforms.set(data.geom_xpos.subarray(id*3,id*3+3),i*12);
  transforms.set(data.geom_xmat.subarray(id*9,id*9+9),i*12+3);
 }
 const values=Object.fromEntries(joints.map(j=>[j.name,data.qpos[j.address]*180/Math.PI]));
 postMessage({type:'pose',time,transforms,values},[transforms.buffer]);
}
self.onmessage=async ({data:message})=>{
 try {
  if(message.type==='seek') { if (data) sendPose(message.time); return; }
  if(message.type!=='init') return;
  status('Загрузка MuJoCo 3.12.0…');
  mj=await loadMujoco({locateFile: name => name.endsWith('.wasm') ? wasmUrl : name});
  status('Загрузка модели G1…');
  mj.FS.mkdir('/model'); mj.FS.mkdir('/model/meshes');
  const base=new URL(message.lite?'model-lite/':'model/',message.base);
  const manifest=JSON.parse(new TextDecoder().decode(await fetchBytes(new URL('manifest.json',base))));
  const mode=message.format==='mjb'?'mjb':'mjcf';
  const files=mode==='mjb'?[manifest.binary||'g1.mjb']:manifest.files;
  // Bounded concurrency reduces transient mesh download memory.
  let cursor=0, done=0;
  await Promise.all(Array.from({length:4},async()=>{
   while(cursor<files.length) {
    const file=files[cursor++];
    let bytes=await fetchBytes(new URL(file,base));
    if(file.endsWith('.gz'))bytes=new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
    mj.FS.writeFile(`/model/${file.replace(/\.gz$/,'')}`,bytes); done++;
    status(`Модель G1: ${done}/${files.length}`);
   }
  }));
  status(mode==='mjb'?'Чтение MJB…':'Компиляция MJCF…');
  const start=performance.now();
  model=mode==='mjb'?mj.MjModel.from_binary_path('/model/g1.mjb',new mj.MjVFS()):mj.MjModel.from_xml_path('/model/g1.xml');
  data=new mj.MjData(model);
  joints=[];
  for(let id=0;id<model.njnt;id++) {
   if(model.jnt_type[id]!==3) continue;
   const name=mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT.value,id);
   if(name) joints.push({name,address:model.jnt_qposadr[id]});
  }
  const initial=Object.fromEntries(joints.map(j=>[j.name,model.qpos0[j.address]]));
  timeline=compile(sample91.program,initial);
  mj.mj_forward(model,data);
  visible=[]; const geometries=[];
  for(let id=0;id<model.ngeom;id++) {
   if(model.geom_group[id]!==1) continue; // official Unitree visual geoms; exclude collision duplicates
   if(model.geom_type[id]!==7) throw new Error(`Unsupported visual geom type ${model.geom_type[id]}`);
   const mesh=model.geom_dataid[id], va=model.mesh_vertadr[mesh], vn=model.mesh_vertnum[mesh];
   const fa=model.mesh_faceadr[mesh],fn=model.mesh_facenum[mesh];
   visible.push(id);
   geometries.push({vertices:Array.from(model.mesh_vert.subarray(va*3,(va+vn)*3)),faces:Array.from(model.mesh_face.subarray(fa*3,(fa+fn)*3)),rgba:Array.from(model.geom_rgba.subarray(id*4,id*4+4))});
  }
  // Files are no longer needed once the model owns its compiled geometry.
  for(const file of files) mj.FS.unlink(`/model/${file.replace(/\.gz$/,'')}`);
  const bounds={min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]};
  // Fit the entire prescribed motion, including extended arms, before playback.
  for(let step=0;step<=60;step++) {
   data.qpos.set(model.qpos0);const pose=poseAt(timeline,timeline.duration*step/60);
   for(const joint of joints)data.qpos[joint.address]=pose[joint.name];
   mj.mj_forward(model,data);
   for(const id of visible)for(let axis=0;axis<3;axis++) {
    const p=data.geom_xpos[id*3+axis],r=model.geom_rbound[id];
    bounds.min[axis]=Math.min(bounds.min[axis],p-r);bounds.max[axis]=Math.max(bounds.max[axis],p+r);
   }
  }
  postMessage({type:'ready',bounds,geometries,duration:timeline.duration,format:mode,compileMs:performance.now()-start,nq:model.nq,njnt:model.njnt});
  sendPose(0);
 } catch(error) { postMessage({type:'error',text:error?.message||String(error)}); }
};
