// Pre-allocated observation helpers — all compute() methods reuse buffers to
// avoid per-frame Float32Array allocations (critical for Safari GC performance).
// Math from utils/math.js is inlined to eliminate intermediate array creation.

class BootIndicator {
  constructor() {
    this._out = new Float32Array(1);
  }

  get size() {
    return 1;
  }

  compute() {
    this._out[0] = 0.0;
    return this._out;
  }
}

class RootAngVelB {
  constructor() {
    this._out = new Float32Array(3);
  }

  get size() {
    return 3;
  }

  compute(state) {
    this._out[0] = state.rootAngVel[0];
    this._out[1] = state.rootAngVel[1];
    this._out[2] = state.rootAngVel[2];
    return this._out;
  }
}

class ProjectedGravityB {
  constructor() {
    this._out = new Float32Array(3);
  }

  get size() {
    return 3;
  }

  compute(state) {
    // Inline inverse-quaternion rotation of gravity=[0,0,-1]
    const qw = state.rootQuat[0], qx = state.rootQuat[1], qy = state.rootQuat[2], qz = state.rootQuat[3];
    const gz = -1;
    const tx = 2 * (qy * gz);
    const ty = 2 * (-qx * gz);
    const tz = 0;
    this._out[0] = -qw * tx + (qy * tz - qz * ty);
    this._out[1] = -qw * ty + (qz * tx - qx * tz);
    this._out[2] = gz - qw * tz + (qx * ty - qy * tx);
    return this._out;
  }
}

class JointPos {
  constructor(policy, kwargs = {}) {
    const { pos_steps = [0, 1, 2, 3, 4, 8] } = kwargs;
    this.posSteps = pos_steps.slice();
    this.numJoints = policy.numActions;

    this.maxStep = Math.max(...this.posSteps);
    this.history = Array.from({ length: this.maxStep + 1 }, () => new Float32Array(this.numJoints));
    this._out = new Float32Array(this.posSteps.length * this.numJoints);
  }

  get size() {
    return this.posSteps.length * this.numJoints;
  }

  reset(state) {
    const source = state?.jointPos ?? new Float32Array(this.numJoints);
    this.history[0].set(source);
    for (let i = 1; i < this.history.length; i++) {
      this.history[i].set(this.history[0]);
    }
  }

  update(state) {
    for (let i = this.history.length - 1; i > 0; i--) {
      this.history[i].set(this.history[i - 1]);
    }
    this.history[0].set(state.jointPos);
  }

  compute() {
    let offset = 0;
    for (const step of this.posSteps) {
      const idx = Math.min(step, this.history.length - 1);
      this._out.set(this.history[idx], offset);
      offset += this.numJoints;
    }
    return this._out;
  }
}

class TrackingCommandObsRaw {
  constructor(policy, kwargs = {}) {
    this.policy = policy;
    this.futureSteps = kwargs.future_steps ?? [0, 2, 4, 8, 16];
    const nFut = this.futureSteps.length;
    this.outputLength = (nFut - 1) * 3 + nFut * 6;
    this._out = new Float32Array(this.outputLength);
    this._indices = new Int32Array(nFut);
  }

  get size() {
    return this.outputLength;
  }

  compute(state) {
    const tracking = this.policy.tracking;
    if (!tracking || !tracking.isReady()) {
      this._out.fill(0);
      return this._out;
    }

    const baseIdx = tracking.refIdx;
    const refLen = tracking.refLen;

    // Inline clampFutureIndices
    const indices = this._indices;
    for (let i = 0; i < this.futureSteps.length; i++) {
      const idx = baseIdx + this.futureSteps[i];
      indices[i] = idx < 0 ? 0 : (idx >= refLen ? refLen - 1 : idx);
    }

    // Inline normalizeQuat for baseQuat
    const bq = tracking.refRootQuat[indices[0]];
    let bw = bq[0], bx = bq[1], by = bq[2], bz = bq[3];
    let n = Math.hypot(bw, bx, by, bz);
    if (n < 1e-9) { bw = 1; bx = by = bz = 0; }
    else { const inv = 1.0 / n; bw *= inv; bx *= inv; by *= inv; bz *= inv; }

    const basePos = tracking.refRootPos[indices[0]];

    let offset = 0;
    // posDiff: for i=1..nFut-1, compute diff then quatApplyInv(baseQuat, diff)
    for (let i = 1; i < indices.length; i++) {
      const pos = tracking.refRootPos[indices[i]];
      const dx = pos[0] - basePos[0], dy = pos[1] - basePos[1], dz = pos[2] - basePos[2];
      // Inline quatApplyInv(baseQuat, [dx, dy, dz])
      const tx = 2 * (by * dz - bz * dy);
      const ty = 2 * (bz * dx - bx * dz);
      const tz = 2 * (bx * dy - by * dx);
      this._out[offset++] = dx - bw * tx + (by * tz - bz * ty);
      this._out[offset++] = dy - bw * ty + (bz * tx - bx * tz);
      this._out[offset++] = dz - bw * tz + (bx * ty - by * tx);
    }

    // Inline normalizeQuat for current root quat
    const sq = state.rootQuat;
    let cw = sq[0], cx = sq[1], cy = sq[2], cz = sq[3];
    n = Math.hypot(cw, cx, cy, cz);
    if (n < 1e-9) { cw = 1; cx = cy = cz = 0; }
    else { const inv = 1.0 / n; cw *= inv; cx *= inv; cy *= inv; cz *= inv; }

    // quatInverse of normalized quat = conjugate
    const ciw = cw, cix = -cx, ciy = -cy, ciz = -cz;

    // rot6d: for each index, normalizeQuat(refQuat), quatMultiply(qCurInv, refQuat), quatToRot6d
    for (let i = 0; i < indices.length; i++) {
      const rq = tracking.refRootQuat[indices[i]];
      // Inline normalizeQuat for refQuat
      let rw = rq[0], rx = rq[1], ry = rq[2], rz = rq[3];
      n = Math.hypot(rw, rx, ry, rz);
      if (n < 1e-9) { rw = 1; rx = ry = rz = 0; }
      else { const inv = 1.0 / n; rw *= inv; rx *= inv; ry *= inv; rz *= inv; }

      // Inline quatMultiply(qCurInv, refQuat)
      const mw = ciw * rw - cix * rx - ciy * ry - ciz * rz;
      const mx = ciw * rx + cix * rw + ciy * rz - ciz * ry;
      const my = ciw * ry - cix * rz + ciy * rw + ciz * rx;
      const mz = ciw * rz + cix * ry - ciy * rx + ciz * rw;

      // Inline quatToRot6d (re-normalize for safety)
      n = Math.hypot(mw, mx, my, mz);
      let nw, nx, ny, nz;
      if (n < 1e-9) { nw = 1; nx = ny = nz = 0; }
      else { const inv = 1.0 / n; nw = mw * inv; nx = mx * inv; ny = my * inv; nz = mz * inv; }

      const xx = nx * nx, yy = ny * ny, zz = nz * nz;
      const xy = nx * ny, xz = nx * nz, yz = ny * nz;
      const wx = nw * nx, wy = nw * ny, wz = nw * nz;

      this._out[offset++] = 1.0 - 2.0 * (yy + zz); // r00
      this._out[offset++] = 2.0 * (xy + wz);         // r10
      this._out[offset++] = 2.0 * (xz - wy);         // r20
      this._out[offset++] = 2.0 * (xy - wz);         // r01
      this._out[offset++] = 1.0 - 2.0 * (xx + zz);   // r11
      this._out[offset++] = 2.0 * (yz + wx);         // r21
    }

    return this._out;
  }
}

class TargetRootZObs {
  constructor(policy, kwargs = {}) {
    this.policy = policy;
    this.futureSteps = kwargs.future_steps ?? [0, 2, 4, 8, 16];
    this._out = new Float32Array(this.futureSteps.length);
    this._indices = new Int32Array(this.futureSteps.length);
  }

  get size() {
    return this.futureSteps.length;
  }

  compute() {
    const tracking = this.policy.tracking;
    if (!tracking || !tracking.isReady()) {
      this._out.fill(0);
      return this._out;
    }

    // Inline clampFutureIndices
    const baseIdx = tracking.refIdx;
    const refLen = tracking.refLen;
    const indices = this._indices;
    for (let i = 0; i < this.futureSteps.length; i++) {
      const idx = baseIdx + this.futureSteps[i];
      indices[i] = idx < 0 ? 0 : (idx >= refLen ? refLen - 1 : idx);
    }

    for (let i = 0; i < indices.length; i++) {
      this._out[i] = tracking.refRootPos[indices[i]][2] + 0.035;
    }
    return this._out;
  }
}

class TargetJointPosObs {
  constructor(policy, kwargs = {}) {
    this.policy = policy;
    this.futureSteps = kwargs.future_steps ?? [0, 2, 4, 8, 16];
    this._indices = new Int32Array(this.futureSteps.length);
    this._out = null;
    this._outLen = 0;
  }

  get size() {
    const nJoints = this.policy.tracking?.nJoints ?? 0;
    return this.futureSteps.length * nJoints;
  }

  compute() {
    const tracking = this.policy.tracking;
    if (!tracking || !tracking.isReady()) {
      const sz = this.size;
      if (!this._out || this._out.length !== sz) { this._out = new Float32Array(sz); this._outLen = sz; }
      this._out.fill(0);
      return this._out;
    }

    const nJoints = tracking.nJoints;
    const totalLen = this.futureSteps.length * nJoints;
    if (!this._out || this._out.length !== totalLen) {
      this._out = new Float32Array(totalLen);
      this._outLen = totalLen;
    }

    // Inline clampFutureIndices
    const baseIdx = tracking.refIdx;
    const refLen = tracking.refLen;
    const indices = this._indices;
    for (let i = 0; i < this.futureSteps.length; i++) {
      const idx = baseIdx + this.futureSteps[i];
      indices[i] = idx < 0 ? 0 : (idx >= refLen ? refLen - 1 : idx);
    }

    let offset = 0;
    for (let i = 0; i < indices.length; i++) {
      this._out.set(tracking.refJointPos[indices[i]], offset);
      offset += nJoints;
    }
    return this._out;
  }
}

class TargetProjectedGravityBObs {
  constructor(policy, kwargs = {}) {
    this.policy = policy;
    this.futureSteps = kwargs.future_steps ?? [0, 2, 4, 8, 16];
    this._out = new Float32Array(this.futureSteps.length * 3);
    this._indices = new Int32Array(this.futureSteps.length);
  }

  get size() {
    return this.futureSteps.length * 3;
  }

  compute() {
    const tracking = this.policy.tracking;
    if (!tracking || !tracking.isReady()) {
      this._out.fill(0);
      return this._out;
    }

    // Inline clampFutureIndices
    const baseIdx = tracking.refIdx;
    const refLen = tracking.refLen;
    const indices = this._indices;
    for (let i = 0; i < this.futureSteps.length; i++) {
      const idx = baseIdx + this.futureSteps[i];
      indices[i] = idx < 0 ? 0 : (idx >= refLen ? refLen - 1 : idx);
    }

    let offset = 0;
    for (let j = 0; j < indices.length; j++) {
      const q = tracking.refRootQuat[indices[j]];
      // Inline normalizeQuat
      let qw = q[0], qx = q[1], qy = q[2], qz = q[3];
      const n = Math.hypot(qw, qx, qy, qz);
      if (n < 1e-9) { qw = 1; qx = qy = qz = 0; }
      else { const inv = 1.0 / n; qw *= inv; qx *= inv; qy *= inv; qz *= inv; }

      // Inline quatApplyInv(quat, [0, 0, -1])
      const gz = -1;
      const tx = 2 * (qy * gz);
      const ty = 2 * (-qx * gz);
      const tz = 0;
      this._out[offset++] = -qw * tx + (qy * tz - qz * ty);
      this._out[offset++] = -qw * ty + (qz * tx - qx * tz);
      this._out[offset++] = gz - qw * tz + (qx * ty - qy * tx);
    }
    return this._out;
  }
}


class PrevActions {
  constructor(policy, kwargs = {}) {
    this.policy = policy;
    const { history_steps = 4 } = kwargs;
    this.steps = Math.max(1, Math.floor(history_steps));
    this.numActions = policy.numActions;
    this.actionBuffer = Array.from({ length: this.steps }, () => new Float32Array(this.numActions));
    this._out = new Float32Array(this.steps * this.numActions);
    this._zeroActions = new Float32Array(this.numActions);
  }

  compute() {
    let offset = 0;
    for (let i = 0; i < this.steps; i++) {
      this._out.set(this.actionBuffer[i], offset);
      offset += this.numActions;
    }
    return this._out;
  }

  reset() {
    for (const buffer of this.actionBuffer) {
      buffer.fill(0.0);
    }
  }

  update() {
    for (let i = this.actionBuffer.length - 1; i > 0; i--) {
      this.actionBuffer[i].set(this.actionBuffer[i - 1]);
    }
    const source = this.policy?.lastActions ?? this._zeroActions;
    this.actionBuffer[0].set(source);
  }

  get size() {
    return this.steps * this.numActions;
  }
}


// Export a dictionary of all observation classes
export const Observations = {
  PrevActions,
  BootIndicator,
  RootAngVelB,
  ProjectedGravityB,
  JointPos,
  TrackingCommandObsRaw,
  TargetRootZObs,
  TargetJointPosObs,
  TargetProjectedGravityBObs
};
