import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {gunzipSync} from 'node:zlib';
import loadMujoco from '@mujoco/mujoco';
import {compile,poseAt} from '../src/timeline.js';
import {sample91} from '../src/sample.js';
test('real MuJoCo WASM: MJCF/MJB parity and all sample 91 qpos',async()=>{
 const mj=await loadMujoco();mj.FS.mkdir('/model');mj.FS.mkdir('/model/meshes');
 const base='public/model';const manifest=JSON.parse(fs.readFileSync(path.join(base,'manifest.json')));
 for(const file of manifest.files)mj.FS.writeFile('/model/'+file,fs.readFileSync(path.join(base,file)));
 mj.FS.writeFile('/model/g1.mjb',gunzipSync(fs.readFileSync(path.join(base,manifest.binary))));
 const xml=mj.MjModel.from_xml_path('/model/g1.xml');
 const binary=mj.MjModel.from_binary_path('/model/g1.mjb',new mj.MjVFS());
 assert.equal(xml.nq,binary.nq);
 let previous;
 for(const model of [xml,binary]) {
  const data=new mj.MjData(model);const initial={},addresses={};
  for(let id=0;id<model.njnt;id++)if(model.jnt_type[id]===3){
   const name=mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT.value,id);
   addresses[name]=model.jnt_qposadr[id];initial[name]=model.qpos0[addresses[name]];
  }
  const timeline=compile(sample91.program,initial);
  for(const t of [0,.375,.75,1.125,1.5]) {
   data.qpos.set(model.qpos0);const pose=poseAt(timeline,t);
   for(const [name,value] of Object.entries(pose))data.qpos[addresses[name]]=value;
   mj.mj_forward(model,data);
   assert.ok(Array.from(data.geom_xpos).every(Number.isFinite));
  }
  const changed=new Set(sample91.program[0].frame.map(j=>addresses[j.name]));
  for(let i=0;i<model.nq;i++)if(!changed.has(i))assert.equal(data.qpos[i],model.qpos0[i]);
  for(const joint of sample91.program[0].frame)assert.ok(Math.abs(data.qpos[addresses[joint.name]]-joint.angle*Math.PI/180)<1e-12);
  const position=name=>{const id=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY.value,name);return Array.from(data.xpos.slice(id*3,id*3+3));};
  const left=position('left_wrist_yaw_link'),right=position('right_wrist_yaw_link'),head=position('torso_link');
  assert.ok(left[2]>head[2]+.2,'anatomical left wrist rises above torso');
  assert.ok(right[1]<-.3,'anatomical right wrist extends to robot right (-Y)');
  const result=Array.from(data.geom_xpos);
  if(previous)result.forEach((n,i)=>assert.ok(Math.abs(n-previous[i])<1e-10));
  previous=result;data.delete();model.delete();
 }
});
