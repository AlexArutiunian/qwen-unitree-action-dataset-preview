/** Compile bounded, deterministic prescribed motion. All poses are radians. */
export function compile(program, initial, limits = {}) {
  const { maxFrames = 100000, maxDepth = 32, maxDuration = 86400 } = limits;
  if (!Array.isArray(program)) throw new Error('Motion must be an array');
  const base = { ...initial }, segments = [];
  let pose = { ...base }, duration = 0, visits = 0;
  function walk(blocks, depth) {
    if (depth > maxDepth) throw new Error('Repeat nesting limit exceeded');
    for (const block of blocks) {
      if (++visits > maxFrames) throw new Error('Motion expansion limit exceeded');
      if (!block || typeof block !== 'object') throw new Error('Invalid block');
      if ('repeat' in block) {
        if (!Array.isArray(block.repeat) || !Number.isSafeInteger(block.times) || block.times < 0 || block.times > maxFrames) throw new Error('Invalid repeat');
        for (let i = 0; i < block.times; i++) walk(block.repeat, depth + 1);
        continue;
      }
      if (!Array.isArray(block.frame) || !Number.isFinite(block.duration) || block.duration < 0) throw new Error('Invalid frame or duration');
      const target = { ...pose }, seen = new Set();
      for (const joint of block.frame) {
        if (!joint || !Object.hasOwn(base, joint.name)) throw new Error(`Unknown joint: ${joint?.name}`);
        if (!Number.isFinite(joint.angle) || seen.has(joint.name)) throw new Error(`Invalid/duplicate joint: ${joint.name}`);
        seen.add(joint.name);
        target[joint.name] = joint.angle * Math.PI / 180;
      }
      const end = duration + block.duration;
      if (end > maxDuration) throw new Error('Motion duration limit exceeded');
      segments.push({ start: duration, end, from: pose, to: target });
      pose = target; duration = end;
    }
  }
  walk(program, 0);
  return { initial: base, final: pose, segments, duration };
}
export function poseAt(timeline, time) {
  if (!Number.isFinite(time)) throw new Error('Invalid time');
  if (time < 0) return { ...timeline.initial };
  if (time >= timeline.duration) return { ...timeline.final };
  const list = timeline.segments;
  let lo = 0, hi = list.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (list[mid].end <= time) lo = mid + 1; else hi = mid; }
  const segment = list[lo];
  if (!segment) return { ...timeline.final };
  const a = Math.max(0, Math.min(1, (time - segment.start) / (segment.end - segment.start)));
  const alpha = a * a * (3 - 2 * a);
  return Object.fromEntries(Object.keys(segment.from).map(name => [name, segment.from[name] + (segment.to[name] - segment.from[name]) * alpha]));
}
