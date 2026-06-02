import {
  normalizeQuat,
  quatConjugate,
  quatMultiply,
  yawComponent
} from './utils/math.js';

function vec3(v) {
  return [Number(v[0]), Number(v[1]), Number(v[2])];
}

function quat(v) {
  return normalizeQuat([Number(v[0]), Number(v[1]), Number(v[2]), Number(v[3])]);
}

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function lerp(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t
  ];
}

function lerpQuat(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
    a[3] + (b[3] - a[3]) * t
  ];
}

function norm(v) {
  return Math.hypot(...v);
}

function norm2(v) {
  return Math.hypot(v[0], v[1]);
}

function dot4(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

function yawToQuat(yaw) {
  return [Math.cos(0.5 * yaw), 0, 0, Math.sin(0.5 * yaw)];
}

function pitchQuatFromDeg(pitchDeg) {
  const pitchRad = pitchDeg * Math.PI / 180.0;
  return [Math.cos(0.5 * pitchRad), 0, Math.sin(0.5 * pitchRad), 0];
}

function quatApply(qIn, v) {
  const q = normalizeQuat(qIn);
  const qc = quatConjugate(q);
  const vq = [0, v[0], v[1], v[2]];
  const out = quatMultiply(quatMultiply(q, vq), qc);
  return [out[1], out[2], out[3]];
}

function slerpQuat(q0In, q1In, t) {
  const q0 = normalizeQuat(q0In);
  let q1 = normalizeQuat(q1In);
  let d = dot4(q0, q1);
  if (d < 0) {
    d = -d;
    q1 = q1.map((x) => -x);
  }
  if (d > 0.9995) {
    return normalizeQuat(lerpQuat(q0, q1, t));
  }
  const theta0 = Math.acos(Math.max(-1, Math.min(1, d)));
  const theta = theta0 * t;
  const sinTheta = Math.sin(theta);
  const sinTheta0 = Math.sin(theta0);
  const s0 = Math.cos(theta) - d * sinTheta / sinTheta0;
  const s1 = sinTheta / sinTheta0;
  return normalizeQuat(q0.map((x, i) => s0 * x + s1 * q1[i]));
}

function interpLinear(p0In, p1In, step) {
  const p0 = vec3(p0In);
  const p1 = vec3(p1In);
  const d = norm(sub(p1, p0));
  if (d < 1e-6) {
    return [p0];
  }
  const n = Math.max(Math.trunc(d / step), 1) + 1;
  return Array.from({ length: n }, (_, i) => lerp(p0, p1, n <= 1 ? 0 : i / (n - 1)));
}

function interpSlerp(q0In, q1In, stepRad) {
  const q0 = quat(q0In);
  const q1 = quat(q1In);
  const d = Math.max(-1, Math.min(1, Math.abs(dot4(q0, q1))));
  const ang = Math.acos(d) * 2.0;
  if (ang < 1e-6) {
    return [q0];
  }
  const n = Math.max(Math.trunc(ang / stepRad), 1) + 1;
  return Array.from({ length: n }, (_, i) => slerpQuat(q0, q1, n <= 1 ? 0 : i / (n - 1)));
}

function linspace(n) {
  if (n <= 1) {
    return [0];
  }
  return Array.from({ length: n }, (_, i) => i / (n - 1));
}

function repeatVec(v, n) {
  const x = Array.from(v);
  return Array.from({ length: n }, () => x.slice());
}

function bcastRows(value, n) {
  if (!Array.isArray(value[0])) {
    return repeatVec(value, n);
  }
  if (value.length === 1 && n > 1) {
    return repeatVec(value[0], n);
  }
  return value.map((row) => Array.from(row));
}

function bcastScalar(value, n) {
  if (Array.isArray(value)) {
    if (value.length === 1 && n > 1) {
      return Array.from({ length: n }, () => Number(value[0]));
    }
    return value.map(Number);
  }
  return Array.from({ length: n }, () => Number(value));
}

function alignQuatHemisphere(rows) {
  const out = rows.map((q) => Array.from(q));
  for (let i = 1; i < out.length; i++) {
    if (dot4(out[i - 1], out[i]) < 0) {
      out[i] = out[i].map((x) => -x);
    }
  }
  return out;
}

function calcHands(centerSeq, baseYawQuat, halfWidthSeq) {
  const n = centerSeq.length;
  const yawSeq = Array.isArray(baseYawQuat[0]) ? baseYawQuat : repeatVec(baseYawQuat, n);
  const widthSeq = Array.isArray(halfWidthSeq)
    ? (halfWidthSeq.length === 1 && n > 1 ? Array.from({ length: n }, () => halfWidthSeq[0]) : halfWidthSeq)
    : Array.from({ length: n }, () => halfWidthSeq);
  const left = [];
  const right = [];
  for (let i = 0; i < n; i++) {
    const y = quatApply(yawSeq[i], [0, 1, 0]);
    left.push(add(centerSeq[i], scale(y, widthSeq[i])));
    right.push(sub(centerSeq[i], scale(y, widthSeq[i])));
  }
  return [left, right];
}

function clampVec2(v, maxNorm) {
  const n = norm2(v);
  if (maxNorm <= 0) {
    return [v[0], v[1]];
  }
  const s = Math.min(1, maxNorm / (n + 1e-8));
  return [v[0] * s, v[1] * s];
}

function heightLerp(z, zLow, zHigh, vLow, vHigh) {
  const t = Math.max(0, Math.min(1, (z - zLow) / (zHigh - zLow + 1e-8)));
  return vLow + (vHigh - vLow) * t;
}

function quatToRpyDeg(qIn) {
  const [w, x, y, z] = quat(qIn);
  const sinrCosp = 2.0 * (w * x + y * z);
  const cosrCosp = 1.0 - 2.0 * (x * x + y * y);
  const roll = Math.atan2(sinrCosp, cosrCosp);
  const sinp = Math.max(-1, Math.min(1, 2.0 * (w * y - z * x)));
  const pitch = Math.asin(sinp);
  const sinyCosp = 2.0 * (w * z + x * y);
  const cosyCosp = 1.0 - 2.0 * (y * y + z * z);
  const yaw = Math.atan2(sinyCosp, cosyCosp);
  const k = 180.0 / Math.PI;
  return [roll * k, pitch * k, yaw * k];
}

function yawTilt(qIn) {
  const y = quat(yawComponent(qIn));
  const tilt = quat(quatMultiply(quatConjugate(y), quat(qIn)));
  return [y, tilt];
}

function yawFromXY(v) {
  return quat(yawToQuat(Math.atan2(v[1], v[0])));
}

class TrajBuilder {
  constructor() {
    this.lwP = [];
    this.lwQ = [];
    this.rwP = [];
    this.rwQ = [];
    this.objP = [];
    this.objQ = [];
    this.contact = [];
    this.phase = [];
    this.torsoP = [];
    this.torsoYawQ = [];
    this.torsoPitchDeg = [];
    this.laP = [];
    this.raP = [];
  }

  append(phase, block) {
    const torsoP = bcastRows(block.torsoP, Array.isArray(block.torsoP[0]) ? block.torsoP.length : 1);
    const n = torsoP.length;
    this.lwP.push(...bcastRows(block.lwP, n));
    this.lwQ.push(...bcastRows(block.lwQ, n));
    this.rwP.push(...bcastRows(block.rwP, n));
    this.rwQ.push(...bcastRows(block.rwQ, n));
    this.objP.push(...bcastRows(block.objP, n));
    this.objQ.push(...bcastRows(block.objQ, n));
    this.torsoP.push(...torsoP);
    this.torsoYawQ.push(...bcastRows(block.torsoYawQ, n));
    this.torsoPitchDeg.push(...bcastScalar(block.torsoPitchDeg, n));
    this.laP.push(...bcastRows(block.laP, n));
    this.raP.push(...bcastRows(block.raP, n));
    this.contact.push(...bcastRows(block.contact, n));
    this.phase.push(...Array.from({ length: n }, () => phase));
  }

  pad(phase, contact, count) {
    if (count <= 0 || this.phase.length === 0) {
      return;
    }
    const n = Math.trunc(count);
    this.append(phase, {
      lwP: repeatVec(this.last('lwP'), n),
      lwQ: repeatVec(this.last('lwQ'), n),
      rwP: repeatVec(this.last('rwP'), n),
      rwQ: repeatVec(this.last('rwQ'), n),
      objP: repeatVec(this.last('objP'), n),
      objQ: repeatVec(this.last('objQ'), n),
      torsoP: repeatVec(this.last('torsoP'), n),
      torsoYawQ: repeatVec(this.last('torsoYawQ'), n),
      torsoPitchDeg: Array.from({ length: n }, () => this.last('torsoPitchDeg')),
      laP: repeatVec(this.last('laP'), n),
      raP: repeatVec(this.last('raP'), n),
      contact: repeatVec(contact, n)
    });
  }

  last(name) {
    const map = {
      lwP: this.lwP,
      lwQ: this.lwQ,
      rwP: this.rwP,
      rwQ: this.rwQ,
      objP: this.objP,
      objQ: this.objQ,
      torsoP: this.torsoP,
      torsoYawQ: this.torsoYawQ,
      torsoPitchDeg: this.torsoPitchDeg,
      laP: this.laP,
      raP: this.raP
    };
    const arr = map[name];
    const value = arr[arr.length - 1];
    return Array.isArray(value) ? value.slice() : value;
  }

  finalize() {
    const yawQ = alignQuatHemisphere(this.torsoYawQ);
    const pitchQuat = this.torsoPitchDeg.map((p) => pitchQuatFromDeg(p));
    const torsoQuat = alignQuatHemisphere(yawQ.map((y, i) => quat(quatMultiply(y, pitchQuat[i]))));
    return {
      refPhase: Int32Array.from(this.phase),
      refLeftWristPos: this.lwP.map((x) => Float32Array.from(x)),
      refLeftWristQuat: alignQuatHemisphere(this.lwQ).map((x) => Float32Array.from(x)),
      refRightWristPos: this.rwP.map((x) => Float32Array.from(x)),
      refRightWristQuat: alignQuatHemisphere(this.rwQ).map((x) => Float32Array.from(x)),
      refObjectPos: this.objP.map((x) => Float32Array.from(x)),
      refObjectQuat: alignQuatHemisphere(this.objQ).map((x) => Float32Array.from(x)),
      refContact: this.contact.map((x) => Float32Array.from(x)),
      refTorsoPos: this.torsoP.map((x) => Float32Array.from(x)),
      refTorsoQuat: torsoQuat.map((x) => Float32Array.from(x)),
      refLeftAnklePos: this.laP.map((x) => Float32Array.from(x)),
      refLeftAnkleQuat: yawQ.map((x) => Float32Array.from(x)),
      refRightAnklePos: this.raP.map((x) => Float32Array.from(x)),
      refRightAnkleQuat: yawQ.map((x) => Float32Array.from(x))
    };
  }
}

export function generateOmniPlanCarryBoxWTACO({
  torsoPos,
  torsoQuat,
  leftAnklePos,
  rightAnklePos,
  objPos,
  objQuat,
  boxHalfDims = [0.2, 0.3, 0.15],
  targetObjPos = [1, 1, 0.5],
  leftWristPos = null,
  leftWristQuat = null,
  rightWristPos = null,
  rightWristQuat = null,
  graspPitchMaxDeg = 20.0,
  placeTorsoFollowObjZ = 0.6
}) {
  const pad = 30;
  const stepLinear = 0.02;
  const stepAngular = 0.03;
  const cfg = {
    handOffsetY: 0.22,
    handOffsetZ: -0.13,
    preGraspDist: 0.4,
    openDelta: 0.3,
    graspMargin: 0.02,
    graspMinWidth: 0.1
  };
  const nominalBoxHalfDims = [0.15, 0.15, 0.15];
  const contact0 = [0, 0, 0, 0];
  const contactHands = [0, 0, 1, 1];
  const rotPitch90 = [0.7017, 0, 0.7017, 0];
  const crouchObjZLow = 0.15;
  const crouchObjZHigh = 0.9;
  const phase14PitchDegLow = 60.0;
  const phase25PitchDegLow = 45.0;
  const crouchXYToObjRatioLow = 1.0;
  const crouchXYMaxShift = 0.1;
  const preplanBentPadFrames = 100;
  const preplanBentPitchThreshDeg = 15.0;
  const preplanBentRollThreshDeg = 15.0;

  const torsoPos0 = vec3(torsoPos);
  const torsoQuat0 = quat(torsoQuat);
  const objPos0 = vec3(objPos);
  const objQuat0 = quat(objQuat);
  const target = vec3(targetObjPos);
  const dims = vec3(boxHalfDims);
  const yaw0 = quat(yawComponent(torsoQuat0));

  const ax = quatApply(objQuat0, [1, 0, 0]);
  const ay = quatApply(objQuat0, [0, 1, 0]);
  const candidates = [
    { pos: add(objPos0, scale(ax, dims[0])), normal: ax, width: dims[1] },
    { pos: sub(objPos0, scale(ax, dims[0])), normal: scale(ax, -1), width: dims[1] },
    { pos: add(objPos0, scale(ay, dims[1])), normal: ay, width: dims[0] },
    { pos: sub(objPos0, scale(ay, dims[1])), normal: scale(ay, -1), width: dims[0] }
  ];
  candidates.sort((a, b) => norm(sub(a.pos, [torsoPos0[0], torsoPos0[1], 0])) - norm(sub(b.pos, [torsoPos0[0], torsoPos0[1], 0])));
  const face = candidates[0];
  const appDir = scale(face.normal, -1);
  const targetYaw = Math.atan2(appDir[1], appDir[0]);
  const targetYawQuat = quat(yawToQuat(targetYaw));

  const planarScale = Math.max(0.8, Math.min(1.5, Math.max(dims[0], dims[1]) / Math.max(nominalBoxHalfDims[0], nominalBoxHalfDims[1])));
  const heightScale = Math.max(0.8, Math.min(1.5, dims[2] / nominalBoxHalfDims[2]));
  const widthScale = Math.max(0.7, Math.min(1.6, face.width / nominalBoxHalfDims[0]));
  const graspParams = {
    handOffsetY: cfg.handOffsetY,
    handOffsetZ: cfg.handOffsetZ * heightScale,
    preGraspDist: cfg.preGraspDist * planarScale,
    graspMargin: cfg.graspMargin * widthScale,
    graspMinWidth: cfg.graspMinWidth
  };
  const zMin = 0.15;
  const zMax = 0.9;
  const graspT = Math.max(0, Math.min(1, (objPos0[2] - zMin) / (zMax - zMin)));
  const graspRelRot = quat(lerpQuat([0.717, 0, 0.717, 0], [1, 0, 0, 0], graspT));
  const qGrasp = quat(quatMultiply(targetYawQuat, graspRelRot));
  const qPregrasp = quat(quatMultiply(targetYawQuat, rotPitch90));
  const appYawQuat = quat(yawToQuat(Math.atan2(objPos0[1] - torsoPos0[1], objPos0[0] - torsoPos0[0])));
  const leftWristQuat0 = leftWristQuat && rightWristQuat ? quat(leftWristQuat) : quat(quatMultiply(yaw0, rotPitch90));
  const rightWristQuat0 = leftWristQuat && rightWristQuat ? quat(rightWristQuat) : leftWristQuat0.slice();

  const b = new TrajBuilder();
  const last = (name) => b.last(name);
  const append = (phase, block) => b.append(phase, block);
  const padPhase = (phase, contact, count) => b.pad(phase, contact, count);

  const phase1112TorsoZ = 0.793;
  const stanceHalfWidth = 0.12;
  const stanceZRel = -0.75;
  const offLA11 = [0, stanceHalfWidth, stanceZRel];
  const offRA11 = [0, -stanceHalfWidth, stanceZRel];
  const offLW0 = [0, graspParams.handOffsetY, graspParams.handOffsetZ];
  const offRW0 = [0, -graspParams.handOffsetY, graspParams.handOffsetZ];
  const [roll0Deg, pitch0Deg] = quatToRpyDeg(torsoQuat0);
  const isBent = Math.abs(pitch0Deg) >= preplanBentPitchThreshDeg || Math.abs(roll0Deg) >= preplanBentRollThreshDeg;
  if (isBent && preplanBentPadFrames > 0) {
    const nPre = preplanBentPadFrames;
    const torsoPre = [torsoPos0[0], torsoPos0[1], phase1112TorsoZ];
    append(10, {
      lwP: repeatVec(add(torsoPre, quatApply(yaw0, offLW0)), nPre),
      lwQ: repeatVec(leftWristQuat0, nPre),
      rwP: repeatVec(add(torsoPre, quatApply(yaw0, offRW0)), nPre),
      rwQ: repeatVec(rightWristQuat0, nPre),
      objP: repeatVec(objPos0, nPre),
      objQ: repeatVec(objQuat0, nPre),
      torsoP: repeatVec(torsoPre, nPre),
      torsoYawQ: repeatVec(yaw0, nPre),
      torsoPitchDeg: Array.from({ length: nPre }, () => 0),
      laP: repeatVec(add(torsoPre, quatApply(yaw0, offLA11)), nPre),
      raP: repeatVec(add(torsoPre, quatApply(yaw0, offRA11)), nPre),
      contact: repeatVec(contact0, nPre)
    });
  }

  const yaw11 = interpSlerp(yaw0, appYawQuat, stepAngular);
  const n11 = yaw11.length;
  const torsoP11 = repeatVec([torsoPos0[0], torsoPos0[1], phase1112TorsoZ], n11);
  const la11 = yaw11.map((q, i) => add(torsoP11[i], quatApply(q, offLA11)));
  const ra11 = yaw11.map((q, i) => add(torsoP11[i], quatApply(q, offRA11)));
  const lwP11 = yaw11.map((q, i) => add(torsoP11[i], quatApply(q, offLW0)));
  const rwP11 = yaw11.map((q, i) => add(torsoP11[i], quatApply(q, offRW0)));
  const u11 = linspace(n11);
  const lwQ11 = yaw11.map((q, i) => slerpQuat(leftWristQuat0, quat(quatMultiply(q, rotPitch90)), u11[i]));
  const rwQ11 = yaw11.map((q, i) => slerpQuat(rightWristQuat0, quat(quatMultiply(q, rotPitch90)), u11[i]));
  append(11, {
    lwP: lwP11, lwQ: alignQuatHemisphere(lwQ11), rwP: rwP11, rwQ: alignQuatHemisphere(rwQ11),
    objP: repeatVec(objPos0, n11), objQ: repeatVec(objQuat0, n11), torsoP: torsoP11,
    torsoYawQ: alignQuatHemisphere(yaw11), torsoPitchDeg: Array.from({ length: n11 }, () => 0),
    laP: la11, raP: ra11, contact: repeatVec(contact0, n11)
  });

  let torsoP120 = last('torsoP');
  let yaw12 = last('torsoYawQ');
  const standoff = 0.7;
  const fwd = quatApply(yaw12, [1, 0, 0]);
  const torsoP121 = torsoP120.slice();
  torsoP121[0] = objPos0[0] - fwd[0] * standoff;
  torsoP121[1] = objPos0[1] - fwd[1] * standoff;
  if (norm2(sub(objPos0, torsoP120)) >= standoff) {
    let torsoP12 = interpLinear(torsoP120, torsoP121, stepLinear);
    if (torsoP12.length < 2) torsoP12 = [torsoP120, torsoP121];
    torsoP12 = torsoP12.map((p) => [p[0], p[1], phase1112TorsoZ]);
    const n12 = torsoP12.length;
    const la12 = torsoP12.map((p) => add(p, quatApply(yaw12, offLA11)));
    const ra12 = torsoP12.map((p) => add(p, quatApply(yaw12, offRA11)));
    const lwP12 = torsoP12.map((p) => add(p, quatApply(yaw12, [0, graspParams.handOffsetY, graspParams.handOffsetZ])));
    const rwP12 = torsoP12.map((p) => add(p, quatApply(yaw12, [0, -graspParams.handOffsetY, graspParams.handOffsetZ])));
    const q12 = quat(quatMultiply(yaw12, rotPitch90));
    append(12, {
      lwP: lwP12, lwQ: repeatVec(q12, n12), rwP: rwP12, rwQ: repeatVec(q12, n12),
      objP: repeatVec(objPos0, n12), objQ: repeatVec(objQuat0, n12), torsoP: torsoP12,
      torsoYawQ: repeatVec(yaw12, n12), torsoPitchDeg: Array.from({ length: n12 }, () => 0),
      laP: la12, raP: ra12, contact: repeatVec(contact0, n12)
    });
  }

  const torsoP13Start = last('torsoP');
  const yaw130 = last('torsoYawQ');
  const lw130 = last('lwP');
  const rw130 = last('rwP');
  const standoff13 = 0.5;
  const fwd13 = quatApply(targetYawQuat, [1, 0, 0]);
  const torsoP13Target = [objPos0[0] - fwd13[0] * standoff13, objPos0[1] - fwd13[1] * standoff13, torsoP13Start[2]];
  const n13 = Math.max(interpSlerp(yaw130, targetYawQuat, stepAngular).length, interpLinear(torsoP13Start, torsoP13Target, 0.005).length, 2);
  const u13 = linspace(n13);
  const yaw13 = u13.map((u) => slerpQuat(yaw130, targetYawQuat, u));
  const torsoP13 = u13.map((u) => lerp(torsoP13Start, torsoP13Target, u));
  const la13 = torsoP13.map((p, i) => add(p, quatApply(yaw13[i], offLA11)));
  const ra13 = torsoP13.map((p, i) => add(p, quatApply(yaw13[i], offRA11)));
  const preCenter = sub(objPos0, scale(quatApply(targetYawQuat, [1, 0, 0]), graspParams.preGraspDist));
  preCenter[2] = Math.max(preCenter[2], 0.7);
  const center130 = scale(add(lw130, rw130), 0.5);
  const halfW13 = 0.5 * norm(sub(lw130, rw130));
  const centers13 = u13.map((u) => lerp(center130, preCenter, u));
  const [lwP13, rwP13] = calcHands(centers13, yaw13, Array.from({ length: n13 }, () => halfW13));
  const lwQ13 = yaw13.map((q) => quat(quatMultiply(q, rotPitch90)));
  append(13, {
    lwP: lwP13, lwQ: alignQuatHemisphere(lwQ13), rwP: rwP13, rwQ: alignQuatHemisphere(lwQ13),
    objP: repeatVec(objPos0, n13), objQ: repeatVec(objQuat0, n13), torsoP: torsoP13,
    torsoYawQ: alignQuatHemisphere(yaw13), torsoPitchDeg: Array.from({ length: n13 }, () => 0),
    laP: la13, raP: ra13, contact: repeatVec(contact0, n13)
  });
  const phase13TorsoPos = last('torsoP');
  const phase13YawQuat = last('torsoYawQ');

  const torsoP140 = last('torsoP');
  const yaw14 = last('torsoYawQ');
  const la140 = last('laP');
  const ra140 = last('raP');
  const objXYTarget = torsoP140.slice();
  const clamp = clampVec2([objPos0[0] - torsoP140[0], objPos0[1] - torsoP140[1]], crouchXYMaxShift);
  objXYTarget[0] = torsoP140[0] + clamp[0];
  objXYTarget[1] = torsoP140[1] + clamp[1];
  const xyRatio14 = heightLerp(objPos0[2], crouchObjZLow, crouchObjZHigh, crouchXYToObjRatioLow, 0);
  const pitchEnd14 = heightLerp(objPos0[2], crouchObjZLow, crouchObjZHigh, phase14PitchDegLow, graspPitchMaxDeg);
  const torsoZStand = Math.max(torsoPos0[2], phase1112TorsoZ);
  const torsoZCrouch = torsoZStand - Math.max(0.6 - objPos0[2], 0);
  const n14 = Math.trunc(60 * xyRatio14);
  const u14 = linspace(n14);
  const torsoP14 = u14.map((u) => {
    const w = xyRatio14 * u;
    return [
      (1 - w) * torsoP140[0] + w * objXYTarget[0],
      (1 - w) * torsoP140[1] + w * objXYTarget[1],
      (1 - u) * torsoZStand + u * torsoZCrouch
    ];
  });
  const centers14 = u14.map((u) => lerp(preCenter, objPos0, u));
  const lwQ14 = u14.map((u) => slerpQuat(qPregrasp, qGrasp, u));
  const yaw14Seq = repeatVec(yaw14, n14);
  const wOpen = graspParams.handOffsetY;
  const wPre = wOpen + 0.1;
  const [lwP14, rwP14] = calcHands(centers14, yaw14Seq, u14.map((u) => wOpen * (1 - u) + wPre * u));
  append(14, {
    lwP: lwP14, lwQ: lwQ14, rwP: rwP14, rwQ: lwQ14, objP: repeatVec(objPos0, n14),
    objQ: repeatVec(objQuat0, n14), torsoP: torsoP14, torsoYawQ: yaw14Seq,
    torsoPitchDeg: u14.map((u) => u * pitchEnd14), laP: repeatVec(la140, n14),
    raP: repeatVec(ra140, n14), contact: repeatVec(contact0, n14)
  });
  padPhase(14, contact0, pad);

  const torsoP150 = last('torsoP');
  const yaw150 = last('torsoYawQ');
  const pitch150 = last('torsoPitchDeg');
  const la150 = last('laP');
  const ra150 = last('raP');
  const wClosed = Math.max(face.width - graspParams.graspMargin, graspParams.graspMinWidth);
  const ws = interpLinear([wPre, 0, 0], [wClosed, 0, 0], 0.01).map((p) => p[0]);
  const n15 = ws.length;
  const centers15 = repeatVec(objPos0, n15);
  const [lwP15, rwP15] = calcHands(centers15, repeatVec(targetYawQuat, n15), ws);
  const contact15 = Array.from({ length: n15 }, (_, i) => i >= Math.floor(n15 / 2) ? contactHands.slice() : contact0.slice());
  append(15, {
    lwP: lwP15, lwQ: repeatVec(qGrasp, n15), rwP: rwP15, rwQ: repeatVec(qGrasp, n15),
    objP: repeatVec(objPos0, n15), objQ: repeatVec(objQuat0, n15), torsoP: repeatVec(torsoP150, n15),
    torsoYawQ: repeatVec(yaw150, n15), torsoPitchDeg: Array.from({ length: n15 }, () => pitch150),
    laP: repeatVec(la150, n15), raP: repeatVec(ra150, n15), contact: contact15
  });
  padPhase(15, contactHands, pad);

  const torsoP210 = last('torsoP');
  const yaw210 = last('torsoYawQ');
  const pitch210 = last('torsoPitchDeg');
  const la210 = last('laP');
  const ra210 = last('raP');
  const lw210 = last('lwP');
  const rw210 = last('rwP');
  const o210 = last('objP');
  const lwq210 = last('lwQ');
  const rwq210 = last('rwQ');
  const oq210 = last('objQ');
  const torsoP211 = [phase13TorsoPos[0], phase13TorsoPos[1], torsoZStand];
  const n21 = Math.trunc(60 * xyRatio14);
  const u21 = linspace(n21);
  const torsoP21 = u21.map((u) => lerp(torsoP210, torsoP211, u));
  const yaw21 = u21.map((u) => slerpQuat(yaw210, phase13YawQuat, u));
  const pitch21 = u21.map((u) => (1 - u) * pitch210);
  const invYaw210 = quatConjugate(yaw21[0]);
  const offLW = quatApply(invYaw210, sub(lw210, torsoP21[0]));
  const offRW = quatApply(invYaw210, sub(rw210, torsoP21[0]));
  const offO = quatApply(invYaw210, sub(o210, torsoP21[0]));
  const [oY0, oTilt0] = yawTilt(oq210);
  const torsoY0 = yaw21[0];
  const invTorsoY0 = quatConjugate(torsoY0);
  const relLDes = yawFromXY(offLW);
  const relRDes = yawFromXY(offRW);
  const relYawFromWristX = (wq, fallback) => {
    const xLocal = quatApply(invTorsoY0, quatApply(wq, [1, 0, 0]));
    xLocal[2] = 0;
    return norm2(xLocal) < 1e-6 ? fallback.slice() : yawFromXY(xLocal);
  };
  const relL0 = relYawFromWristX(lwq210, relLDes);
  const relR0 = relYawFromWristX(rwq210, relRDes);
  const lwYaw0 = quat(quatMultiply(torsoY0, relL0));
  const rwYaw0 = quat(quatMultiply(torsoY0, relR0));
  const lwTilt0 = quat(quatMultiply(quatConjugate(lwYaw0), lwq210));
  const rwTilt0 = quat(quatMultiply(quatConjugate(rwYaw0), rwq210));
  const relOYaw = quat(quatMultiply(quatConjugate(torsoY0), oY0));
  const qIdent = [1, 0, 0, 0];
  const lwP21 = [];
  const rwP21 = [];
  const oP21 = [];
  const lwQ21 = [];
  const rwQ21 = [];
  const oQ21 = [];
  for (let k = 0; k < n21; k++) {
    const u = u21[k];
    const lift = [0, 0, 0.1 * u];
    lwP21.push(add(add(torsoP21[k], quatApply(yaw21[k], offLW)), lift));
    rwP21.push(add(add(torsoP21[k], quatApply(yaw21[k], offRW)), lift));
    oP21.push(add(add(torsoP21[k], quatApply(yaw21[k], offO)), lift));
    const lwY = quat(quatMultiply(yaw21[k], slerpQuat(relL0, relLDes, u)));
    const rwY = quat(quatMultiply(yaw21[k], slerpQuat(relR0, relRDes, u)));
    const oY = quat(quatMultiply(yaw21[k], relOYaw));
    lwQ21.push(quat(quatMultiply(lwY, slerpQuat(lwTilt0, qIdent, u))));
    rwQ21.push(quat(quatMultiply(rwY, slerpQuat(rwTilt0, qIdent, u))));
    oQ21.push(quat(quatMultiply(oY, slerpQuat(oTilt0, qIdent, u))));
  }
  append(21, {
    lwP: lwP21, lwQ: alignQuatHemisphere(lwQ21), rwP: rwP21, rwQ: alignQuatHemisphere(rwQ21),
    objP: oP21, objQ: alignQuatHemisphere(oQ21), torsoP: torsoP21, torsoYawQ: alignQuatHemisphere(yaw21),
    torsoPitchDeg: pitch21, laP: repeatVec(la210, n21), raP: repeatVec(ra210, n21),
    contact: repeatVec(contactHands, n21)
  });
  padPhase(21, contactHands, pad);

  const torsoP230 = last('torsoP');
  const yaw230 = last('torsoYawQ');
  const pitch230 = last('torsoPitchDeg');
  const la230 = last('laP');
  const ra230 = last('raP');
  const lw230 = last('lwP');
  const rw230 = last('rwP');
  const o230 = last('objP');
  const lwq230 = last('lwQ');
  const rwq230 = last('rwQ');
  const oq230 = last('objQ');
  const diff23 = sub(target, o230);
  const yaw231 = norm2(diff23) < 1e-6 ? yaw230 : quat(yawToQuat(Math.atan2(diff23[1], diff23[0])));
  const yaw23 = interpSlerp(yaw230, yaw231, stepAngular);
  const n23 = yaw23.length;
  const invYaw230 = quatConjugate(yaw230);
  const offLA = quatApply(invYaw230, sub(la230, torsoP230));
  const offRA = quatApply(invYaw230, sub(ra230, torsoP230));
  const offLW23 = quatApply(invYaw230, sub(lw230, torsoP230));
  const offRW23 = quatApply(invYaw230, sub(rw230, torsoP230));
  const offO23 = quatApply(invYaw230, sub(o230, torsoP230));
  const [lwY0, lwTilt23] = yawTilt(lwq230);
  const [rwY0, rwTilt23] = yawTilt(rwq230);
  const [oY23, oTilt23] = yawTilt(oq230);
  const relLYaw = quatMultiply(quatConjugate(yaw230), lwY0);
  const relRYaw = quatMultiply(quatConjugate(yaw230), rwY0);
  const relOYaw23 = quatMultiply(quatConjugate(yaw230), oY23);
  const la23 = [];
  const ra23 = [];
  const lwP23 = [];
  const rwP23 = [];
  const oP23 = [];
  const lwQ23 = [];
  const rwQ23 = [];
  const oQ23 = [];
  for (const y of yaw23) {
    la23.push(add(torsoP230, quatApply(y, offLA)));
    ra23.push(add(torsoP230, quatApply(y, offRA)));
    lwP23.push(add(torsoP230, quatApply(y, offLW23)));
    rwP23.push(add(torsoP230, quatApply(y, offRW23)));
    oP23.push(add(torsoP230, quatApply(y, offO23)));
    lwQ23.push(quat(quatMultiply(quat(quatMultiply(y, relLYaw)), lwTilt23)));
    rwQ23.push(quat(quatMultiply(quat(quatMultiply(y, relRYaw)), rwTilt23)));
    oQ23.push(quat(quatMultiply(quat(quatMultiply(y, relOYaw23)), oTilt23)));
  }
  append(23, {
    lwP: lwP23, lwQ: alignQuatHemisphere(lwQ23), rwP: rwP23, rwQ: alignQuatHemisphere(rwQ23),
    objP: oP23, objQ: alignQuatHemisphere(oQ23), torsoP: repeatVec(torsoP230, n23),
    torsoYawQ: alignQuatHemisphere(yaw23), torsoPitchDeg: Array.from({ length: n23 }, () => pitch230),
    laP: la23, raP: ra23, contact: repeatVec(contactHands, n23)
  });

  const torsoP240 = last('torsoP');
  const yaw24 = last('torsoYawQ');
  const pitch24 = last('torsoPitchDeg');
  const la240 = last('laP');
  const ra240 = last('raP');
  const lw240 = last('lwP');
  const rw240 = last('rwP');
  const o240 = last('objP');
  const oq24 = last('objQ');
  const lwq24 = last('lwQ');
  const rwq24 = last('rwQ');
  let moveDir = sub(target, o240);
  moveDir[2] = 0;
  const xyNorm = norm2(moveDir);
  if (xyNorm > 1e-6) moveDir = scale(moveDir, 1 / (xyNorm + 1e-8));
  const finalObj24 = sub(target, scale(moveDir, 0.1));
  finalObj24[2] = o240[2];
  const delta24 = sub(finalObj24, o240);
  delta24[2] = 0;
  const torsoP24 = interpLinear(torsoP240, add(torsoP240, delta24), stepLinear);
  const n24 = torsoP24.length;
  const dseq24 = torsoP24.map((p) => sub(p, torsoP24[0]));
  append(24, {
    lwP: dseq24.map((d) => add(lw240, d)), lwQ: repeatVec(lwq24, n24),
    rwP: dseq24.map((d) => add(rw240, d)), rwQ: repeatVec(rwq24, n24),
    objP: dseq24.map((d) => [o240[0] + d[0], o240[1] + d[1], o240[2]]), objQ: repeatVec(oq24, n24),
    torsoP: torsoP24, torsoYawQ: repeatVec(yaw24, n24), torsoPitchDeg: Array.from({ length: n24 }, () => pitch24),
    laP: dseq24.map((d) => add(la240, d)), raP: dseq24.map((d) => add(ra240, d)),
    contact: repeatVec(contactHands, n24)
  });
  padPhase(24, contactHands, pad);

  const torsoP250 = last('torsoP');
  const yaw25 = last('torsoYawQ');
  const pitch25 = last('torsoPitchDeg');
  const la250 = last('laP');
  const ra250 = last('raP');
  const lw250 = last('lwP');
  const rw250 = last('rwP');
  const o250 = last('objP');
  const oq25 = last('objQ');
  const lwq25 = last('lwQ');
  const rwq25 = last('rwQ');
  const goalZ = Number.isFinite(target[2]) ? target[2] : 0.5;
  const xyRatio25 = heightLerp(goalZ, crouchObjZLow, crouchObjZHigh, crouchXYToObjRatioLow, 0);
  const pitchEnd25 = heightLerp(goalZ, crouchObjZLow, crouchObjZHigh, phase25PitchDegLow, 0);
  const clamp25 = clampVec2([o250[0] - torsoP250[0], o250[1] - torsoP250[1]], crouchXYMaxShift);
  const objXYTarget25 = [torsoP250[0] + clamp25[0], torsoP250[1] + clamp25[1], torsoP250[2]];
  const n25 = Math.trunc(60 * xyRatio25);
  const u25 = linspace(n25);
  const oP25 = u25.map((u) => [o250[0], o250[1], (1 - u) * o250[2] + u * goalZ]);
  const torsoP25 = u25.map((u, i) => [
    (1 - xyRatio25 * u) * torsoP250[0] + xyRatio25 * u * objXYTarget25[0],
    (1 - xyRatio25 * u) * torsoP250[1] + xyRatio25 * u * objXYTarget25[1],
    torsoP250[2] + placeTorsoFollowObjZ * (oP25[i][2] - o250[2])
  ]);
  const offLWO = sub(lw250, o250);
  const offRWO = sub(rw250, o250);
  append(25, {
    lwP: oP25.map((p) => add(p, offLWO)), lwQ: repeatVec(lwq25, n25),
    rwP: oP25.map((p) => add(p, offRWO)), rwQ: repeatVec(rwq25, n25),
    objP: oP25, objQ: repeatVec(oq25, n25), torsoP: torsoP25, torsoYawQ: repeatVec(yaw25, n25),
    torsoPitchDeg: u25.map((u) => (1 - u) * pitch25 + u * pitchEnd25),
    laP: repeatVec(la250, n25), raP: repeatVec(ra250, n25), contact: repeatVec(contactHands, n25)
  });
  padPhase(25, contactHands, pad);

  const torsoP260 = last('torsoP');
  const yaw26 = last('torsoYawQ');
  const pitch26 = last('torsoPitchDeg');
  const la260 = last('laP');
  const ra260 = last('raP');
  const lw260 = last('lwP');
  const rw260 = last('rwP');
  const o260 = last('objP');
  const oq26 = last('objQ');
  const lwq26 = last('lwQ');
  const rwq26 = last('rwQ');
  const n26 = 40;
  const center26 = scale(add(lw260, rw260), 0.5);
  const axis26 = sub(lw260, rw260);
  const dist0 = norm(axis26);
  const axisUnit = scale(axis26, 1 / (dist0 + 1e-8));
  const t26 = linspace(n26);
  const lwP26 = t26.map((t) => add(center26, scale(axisUnit, ((1 - t) * dist0 + t * (dist0 + cfg.openDelta)) * 0.5)));
  const rwP26 = t26.map((t) => sub(center26, scale(axisUnit, ((1 - t) * dist0 + t * (dist0 + cfg.openDelta)) * 0.5)));
  const contact26 = Array.from({ length: n26 }, (_, i) => i >= Math.trunc((n26 + 1) / 2) ? contact0.slice() : contactHands.slice());
  append(26, {
    lwP: lwP26, lwQ: repeatVec(lwq26, n26), rwP: rwP26, rwQ: repeatVec(rwq26, n26),
    objP: repeatVec(o260, n26), objQ: repeatVec(oq26, n26), torsoP: repeatVec(torsoP260, n26),
    torsoYawQ: repeatVec(yaw26, n26), torsoPitchDeg: Array.from({ length: n26 }, () => pitch26),
    laP: repeatVec(la260, n26), raP: repeatVec(ra260, n26), contact: contact26
  });
  padPhase(26, contact0, pad);

  const torsoP270 = last('torsoP');
  const yaw27 = last('torsoYawQ');
  const pitch270 = last('torsoPitchDeg');
  const la270 = last('laP');
  const ra270 = last('raP');
  const lw270 = last('lwP');
  const rw270 = last('rwP');
  const lwq270 = last('lwQ');
  const rwq270 = last('rwQ');
  const o270 = last('objP');
  const oq27 = last('objQ');
  const n27 = Math.max(pad, 60) + 1;
  const u27 = linspace(n27);
  const torsoP271 = [torsoP270[0], torsoP270[1], torsoZStand];
  const torsoP27 = u27.map((u) => lerp(torsoP270, torsoP271, u));
  const invYaw27 = quatConjugate(yaw27);
  const offL0 = quatApply(invYaw27, sub(lw270, torsoP270));
  const offR0 = quatApply(invYaw27, sub(rw270, torsoP270));
  const offL1 = [0, graspParams.handOffsetY, graspParams.handOffsetZ - 0.05];
  const offR1 = [0, -graspParams.handOffsetY, graspParams.handOffsetZ - 0.05];
  let qDown = quat(quatMultiply(yaw27, pitchQuatFromDeg(90)));
  const lwP27 = [];
  const rwP27 = [];
  const lwQ27 = [];
  const rwQ27 = [];
  for (let k = 0; k < n27; k++) {
    const u = u27[k];
    lwP27.push(add(torsoP27[k], quatApply(yaw27, lerp(offL0, offL1, u))));
    rwP27.push(add(torsoP27[k], quatApply(yaw27, lerp(offR0, offR1, u))));
    let lTgt = qDown;
    let rTgt = qDown;
    if (dot4(lwq270, lTgt) < 0) lTgt = lTgt.map((x) => -x);
    if (dot4(rwq270, rTgt) < 0) rTgt = rTgt.map((x) => -x);
    lwQ27.push(slerpQuat(lwq270, lTgt, u));
    rwQ27.push(slerpQuat(rwq270, rTgt, u));
  }
  append(27, {
    lwP: lwP27, lwQ: alignQuatHemisphere(lwQ27), rwP: rwP27, rwQ: alignQuatHemisphere(rwQ27),
    objP: repeatVec(o270, n27), objQ: repeatVec(oq27, n27), torsoP: torsoP27,
    torsoYawQ: repeatVec(yaw27, n27), torsoPitchDeg: u27.map((u) => (1 - u) * pitch270),
    laP: repeatVec(la270, n27), raP: repeatVec(ra270, n27), contact: repeatVec(contact0, n27)
  });
  padPhase(27, contact0, pad);

  const out = b.finalize();
  out.targetYaw = targetYaw;
  return out;
}
