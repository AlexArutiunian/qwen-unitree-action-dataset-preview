import test from 'node:test';
import assert from 'node:assert/strict';
import { compile, poseAt } from '../src/timeline.js';
import { sample91 } from '../src/sample.js';
const frame = (name, angle, duration=1) => ({frame:[{name,angle}],duration});
test('partial updates, hold, radians, arbitrary seeking', () => {
 const t=compile([frame('arm',180),{frame:[],duration:2},frame('finger',90)],{arm:0,finger:0.1});
 assert.equal(t.duration,4); assert.equal(poseAt(t,0.5).arm,Math.PI/2);
 assert.equal(poseAt(t,2).finger,0.1); assert.equal(poseAt(t,2).arm,Math.PI);
 assert.equal(poseAt(t,4).finger,Math.PI/2); assert.equal(poseAt(t,0).finger,0.1);
});
test('nested repeats accumulate from previous pose',()=>{
 const t=compile([{repeat:[frame('a',90),{repeat:[frame('b',30),{frame:[],duration:1}],times:2}],times:3}],{a:0,b:0});
 assert.equal(t.duration,15); assert.equal(t.segments.length,15); assert.equal(poseAt(t,5).a,Math.PI/2);
});
test('instant frames and empty programs',()=>{
 assert.deepEqual(poseAt(compile([],{a:1}),0),{a:1});
 const t=compile([frame('a',90,0),{frame:[],duration:1}],{a:0});
 assert.equal(poseAt(t,0).a,Math.PI/2);
});
test('sample 91 exact anatomical targets; neutral torso and legs preserved',()=>{
 const initial=Object.fromEntries(sample91.program[0].frame.map(j=>[j.name,0]));
 initial.waist=0.2; initial.knee=0.3;
 const t=compile(sample91.program,initial), end=poseAt(t,1.5);
 for (const j of sample91.program[0].frame) assert.equal(end[j.name],j.angle*Math.PI/180);
 assert.equal(end.waist,0.2); assert.equal(end.knee,0.3);
});
test('reject invalid or oversized motion instead of freezing',()=>{
 assert.throws(()=>compile([frame('missing',10)],{a:0}),/Unknown/);
 assert.throws(()=>compile([frame('a',NaN)],{a:0}));
 assert.throws(()=>compile([{repeat:[],times:Infinity}],{}));
 assert.throws(()=>compile([{repeat:[{frame:[],duration:1}],times:100}],{}, {maxFrames:10}));
 assert.throws(()=>compile([frame('a',1,-1)],{a:0}));
});
