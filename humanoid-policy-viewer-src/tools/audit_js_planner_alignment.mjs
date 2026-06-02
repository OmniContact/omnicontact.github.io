#!/usr/bin/env node
import fs from 'node:fs';
import { generateOmniPlanCarryBoxWTACO } from '../src/simulation/omniPlanCarryBoxWTACO.js';

const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const scene = input.scene;
const js = generateOmniPlanCarryBoxWTACO({
  torsoPos: scene.torso_link.pos,
  torsoQuat: scene.torso_link.quat,
  leftAnklePos: scene.left_ankle_pitch_link.pos,
  rightAnklePos: scene.right_ankle_pitch_link.pos,
  objPos: scene.box.pos,
  objQuat: scene.box.quat,
  boxHalfDims: input.dims,
  targetObjPos: input.goal,
  leftWristPos: scene.left_palm_link.pos,
  leftWristQuat: scene.left_palm_link.quat,
  rightWristPos: scene.right_palm_link.pos,
  rightWristQuat: scene.right_palm_link.quat
});

const keyMap = {
  ref_phase: 'refPhase',
  ref_left_wrist_pos: 'refLeftWristPos',
  ref_left_wrist_quat: 'refLeftWristQuat',
  ref_right_wrist_pos: 'refRightWristPos',
  ref_right_wrist_quat: 'refRightWristQuat',
  ref_object_pos: 'refObjectPos',
  ref_object_quat: 'refObjectQuat',
  ref_contact: 'refContact',
  ref_torso_future_pos: 'refTorsoPos',
  ref_torso_future_quat: 'refTorsoQuat',
  ref_left_ankle_future_pos: 'refLeftAnklePos',
  ref_left_ankle_future_quat: 'refLeftAnkleQuat',
  ref_right_ankle_future_pos: 'refRightAnklePos',
  ref_right_ankle_future_quat: 'refRightAnkleQuat'
};

function flattenRows(x) {
  if (!Array.isArray(x)) {
    return Array.from(x);
  }
  if (x.length === 0) {
    return [];
  }
  if (!Array.isArray(x[0]) && !(ArrayBuffer.isView(x[0]))) {
    return Array.from(x);
  }
  return x.flatMap((row) => Array.from(row));
}

function quatAngle(a, b) {
  let dot = 0;
  for (let i = 0; i < 4; i++) dot += a[i] * b[i];
  dot = Math.abs(dot);
  dot = Math.max(-1, Math.min(1, dot));
  return 2 * Math.acos(dot);
}

function rowError(nativeRows, jsRows, isQuat) {
  const n = Math.min(nativeRows.length, jsRows.length);
  const errs = [];
  for (let i = 0; i < n; i++) {
    const a = Array.from(nativeRows[i]);
    const b = Array.from(jsRows[i]);
    if (isQuat) {
      errs.push(quatAngle(a, b));
    } else {
      let s = 0;
      for (let j = 0; j < a.length; j++) {
        const d = a[j] - b[j];
        s += d * d;
      }
      errs.push(Math.sqrt(s));
    }
  }
  return errs;
}

function mean(xs) {
  return xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
}

function phaseCounts(rows) {
  const map = new Map();
  for (const v of rows) map.set(Number(v), (map.get(Number(v)) ?? 0) + 1);
  return Array.from(map.entries()).map(([k, v]) => `${k}:${v}`).join(' ');
}

console.log(`native_len=${input.native.ref_phase.length} js_len=${js.refPhase.length} target_yaw(native/js)=${input.nativeTargetYaw.toFixed(8)}/${js.targetYaw.toFixed(8)}`);
console.log(`native_phase_counts=${phaseCounts(input.native.ref_phase)}`);
console.log(`js_phase_counts    =${phaseCounts(Array.from(js.refPhase))}`);

let failed = false;
for (const [nativeKey, jsKey] of Object.entries(keyMap)) {
  const nativeValue = input.native[nativeKey];
  const jsValue = js[jsKey];
  const nativeLen = nativeValue.length;
  const jsLen = jsValue.length;
  const n = Math.min(nativeLen, jsLen);
  if (nativeKey === 'ref_phase') {
    let mismatch = 0;
    let first = -1;
    for (let i = 0; i < n; i++) {
      if (Number(nativeValue[i]) !== Number(jsValue[i])) {
        mismatch++;
        if (first < 0) first = i;
      }
    }
    if (nativeLen !== jsLen || mismatch) failed = true;
    console.log(`${nativeKey.padEnd(34)} len ${nativeLen}/${jsLen} mismatch=${mismatch} first=${first}`);
    continue;
  }

  const errs = rowError(nativeValue, jsValue, nativeKey.includes('quat'));
  let imax = 0;
  for (let i = 1; i < errs.length; i++) if (errs[i] > errs[imax]) imax = i;
  const maxErr = errs[imax] ?? 0;
  const tolerance = nativeKey.includes('quat') ? 2e-3 : 5e-4;
  if (nativeLen !== jsLen || maxErr > tolerance) failed = true;
  const unit = nativeKey.includes('quat') ? 'rad' : 'L2';
  console.log(`${nativeKey.padEnd(34)} len ${nativeLen}/${jsLen} mean=${mean(errs).toExponential(3)} max=${maxErr.toExponential(3)} ${unit} @${imax}`);
}

if (failed) {
  process.exitCode = 1;
}
