/**
 * InteractionPolicyRunner
 *
 * Orchestrates the full interaction policy pipeline:
 *   1. Read body positions from MuJoCo simulation
 *   2. Compute SDF distances and gradients via voxel grid lookup
 *   3. Transform gradients to each body's local frame
 *   4. Run latent encoder ONNX: SDF(28) -> latent(64)
 *   5. Maintain history buffers (joint_states, gravities, root_ang_vels, last_actions, interaction_latents)
 *   6. Compute full observation (proprio + movement_goal + task_condition + interaction_field)
 *   7. Run policy ONNX inference -> actions
 */
import * as ort from 'onnxruntime-web';
import { SDFManager } from './sdfHelper.js';

// Configure ORT WASM: single-thread avoids SharedArrayBuffer memory growth
ort.env.wasm.numThreads = 1;

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/** FIFO ring buffer that keeps the last `maxSize` entries. */
class HistoryQueue {
  constructor(size, maxSize) {
    this.maxSize = maxSize;
    this.elementSize = size;
    this.buffer = [];
    for (let i = 0; i < maxSize; i++) {
      this.buffer.push(new Float32Array(size));
    }
    this._flat = new Float32Array(maxSize * size);
  }
  push(arr) {
    // Rotate: move oldest to front reuse slot, shift others
    const recycled = this.buffer.pop();
    recycled.set(arr);
    this.buffer.unshift(recycled);
  }
  clear() {
    for (const buf of this.buffer) buf.fill(0.0);
    this._flat.fill(0.0);
  }
  /** Return flattened history (newest first) into pre-allocated buffer. */
  flatten() {
    for (let i = 0; i < this.maxSize; i++) {
      this._flat.set(this.buffer[i], i * this.elementSize);
    }
    return this._flat;
  }
}

// quatToMatrix, matrixFirst2Cols, _quatApplyForward have been inlined
// directly into step() to avoid per-frame array allocations.

/* ------------------------------------------------------------------ */
/*  InteractionPolicyRunner                                            */
/* ------------------------------------------------------------------ */

export class InteractionPolicyRunner {
  /**
   * @param {object} config  – parsed interaction_policy.json
   */
  constructor(config) {
    this.config = config;
    this.numActions = config.policy_joint_names.length;          // 29
    this.numHistory = config.history_length ?? 5;
    this.actionClip = config.action_clip ?? 10.0;
    this.maxValidDistance = config.sdf?.max_valid_distance ?? 1.0;

    this.lastActions = new Float32Array(this.numActions);

    /* ---------- SDF ---------- */
    this.sdfManager = new SDFManager();
    this.sdfManager.maxValidDistance = this.maxValidDistance;
    this.sdfBodyNames = config.sdf?.query_body_names ?? [];
    this.numSdfBodies = this.sdfBodyNames.length;               // 7

    /* ---------- history buffers ---------- */
    const jointStateSize = this.numActions * 2;                  // pos + vel*0.05
    this.histJointStates   = new HistoryQueue(jointStateSize, this.numHistory);
    this.histGravities     = new HistoryQueue(3, this.numHistory);
    this.histRootAngVels   = new HistoryQueue(3, this.numHistory);
    this.histLastActions   = new HistoryQueue(this.numActions, this.numHistory);
    this.histInteractionLatents = new HistoryQueue(64, this.numHistory);

    /* ---------- target root tracking ---------- */
    this.targetRootPos      = new Float32Array(3);
    this.targetRootQuat     = new Float32Array([1, 0, 0, 0]);
    this.prevTargetRootPos  = new Float32Array(3);
    this.taskCondition      = new Float32Array(2);               // 0=none, 1=carry, 2=push

    /* ---------- ONNX sessions ---------- */
    this.policySession = null;
    this.latentEncoderSession = null;
    this.isInferencing = false;

    /* ---------- SDF visualization data (exposed for rendering) ---------- */
    this.sdfVisDistances = new Float32Array(this.numSdfBodies);
    this.sdfVisGradients = new Float32Array(this.numSdfBodies * 3); // world frame
    this.sdfVisPositions = new Float32Array(this.numSdfBodies * 3); // world frame

    /* ---------- Pre-allocated step() buffers (reused each frame) ---------- */
    this._sdfGradLocal = new Float32Array(this.numSdfBodies * 3);
    this._interactionSdf = new Float32Array(this.numSdfBodies + this.numSdfBodies * 3); // dist + grad = 28
    this._latent = new Float32Array(64);
    this._jointState = new Float32Array(this.numActions * 2);
    this._gravLocalBuf = new Float32Array(3);
    this._proprio = new Float32Array(
      this.numActions * 2 * this.numHistory +
      3 * this.numHistory +
      3 * this.numHistory +
      this.numActions * this.numHistory
    );
    this._movementGoal = new Float32Array(12);
    this._target = new Float32Array(this.numActions);

    // All quaternion/vector math is now inlined in step() to avoid
    // per-frame object allocation — no temporary buffers needed.
  }

  /* ============================== Dispose =========================== */

  /**
   * Release ONNX sessions and free WASM memory.
   * Must be called before discarding this runner.
   */
  async dispose() {
    // Dispose reusable input tensors
    if (this._leInputTensor) {
      try { this._leInputTensor.dispose(); } catch (e) { /* ignore */ }
      this._leInputTensor = null;
      this._leInputFeed = null;
    }
    if (this._policyTensors) {
      for (const key in this._policyTensors) {
        try { this._policyTensors[key].dispose(); } catch (e) { /* ignore */ }
      }
      this._policyTensors = null;
      this._policyFeed = null;
    }
    // Release ONNX sessions (frees WASM memory)
    if (this.policySession) {
      try { await this.policySession.release(); } catch (e) { /* ignore */ }
      this.policySession = null;
    }
    if (this.latentEncoderSession) {
      try { await this.latentEncoderSession.release(); } catch (e) { /* ignore */ }
      this.latentEncoderSession = null;
    }
  }

  /* ============================== Init ============================== */

  async init() {
    const onnxCfg = this.config.onnx;
    const leCfg = this.config.latent_encoder;

    // Load policy ONNX
    const policyBuf = await (await fetch(onnxCfg.path)).arrayBuffer();
    this.policySession = await ort.InferenceSession.create(policyBuf, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      enableMemPattern: true,
      enableCpuMemArena: true,
    });
    console.log('Policy ONNX loaded. Inputs:', this.policySession.inputNames,
                'Outputs:', this.policySession.outputNames);

    // Load latent encoder ONNX
    const leBuf = await (await fetch(leCfg.path)).arrayBuffer();
    this.latentEncoderSession = await ort.InferenceSession.create(leBuf, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      enableMemPattern: true,
      enableCpuMemArena: true,
    });
    console.log('Latent encoder ONNX loaded. Inputs:', this.latentEncoderSession.inputNames,
                'Outputs:', this.latentEncoderSession.outputNames);

    // Create reusable ORT input tensors (wraps pre-allocated arrays by reference).
    // ORT reads from the underlying array at session.run() time, so updating
    // array contents between calls works correctly without creating new tensors.
    this._leInputTensor = new ort.Tensor('float32', this._interactionSdf, [1, this._interactionSdf.length]);
    this._leInputFeed = { [this.latentEncoderSession.inputNames[0]]: this._leInputTensor };

    this._policyTensors = {
      'proprio': new ort.Tensor('float32', this._proprio, [1, this._proprio.length]),
      'movement_goal': new ort.Tensor('float32', this._movementGoal, [1, 12]),
      'task_condition': new ort.Tensor('float32', this.taskCondition, [1, 2]),
      'interaction_field': new ort.Tensor('float32', this.histInteractionLatents._flat, [1, this.histInteractionLatents._flat.length]),
    };
    // Build the feed dict mapping session input names to our reusable tensors
    this._policyFeed = {};
    for (const name of this.policySession.inputNames) {
      if (this._policyTensors[name]) {
        this._policyFeed[name] = this._policyTensors[name];
      }
    }

    // Load SDF grids
    const sdfObjects = this.config.sdf?.objects ?? {};
    for (const [name, url] of Object.entries(sdfObjects)) {
      await this.sdfManager.loadGrid(name, url);
    }
  }

  /* ============================== Reset ============================= */

  reset(state) {
    this.lastActions.fill(0.0);
    this.histJointStates.clear();
    this.histGravities.clear();
    this.histRootAngVels.clear();
    this.histLastActions.clear();
    this.histInteractionLatents.clear();
    this.targetRootPos.fill(0.0);
    this.targetRootQuat.set([1, 0, 0, 0]);
    this.prevTargetRootPos.fill(0.0);
    this.taskCondition.fill(0.0);

    if (state) {
      this.targetRootPos[0] = state.rootPos[0];
      this.targetRootPos[1] = state.rootPos[1];
      this.targetRootPos[2] = state.rootPos[2];
      this.prevTargetRootPos.set(this.targetRootPos);
    }
  }

  /* ========================== Main step ============================= */

  /**
   * Run one policy step.
   * @param {object} state
   *   - jointPos: Float32Array(29)
   *   - jointVel: Float32Array(29)
   *   - rootPos:  Float32Array(3)   [x,y,z] world
   *   - rootQuat: Float32Array(4)   [w,x,y,z]
   *   - rootAngVel: Float32Array(3)
   *   - sdfBodyPositions:  Float32Array(7*3) flat [x,y,z, ...]
   *   - sdfBodyQuaternions: Float32Array(7*4) flat [w,x,y,z, ...]
   * @returns {Float32Array|null} joint position targets (numActions)
   */
  async step(state) {
    if (this.isInferencing) return null;
    this.isInferencing = true;

    try {
      /* ---- 1. SDF query ---- */
      const { distances: sdfDist, gradients: sdfGradWorld } =
        this.sdfManager.queryClosest(state.sdfBodyPositions, this.numSdfBodies);

      // Store for visualization (world frame, before local-frame transform)
      this.sdfVisDistances.set(sdfDist);
      this.sdfVisGradients.set(sdfGradWorld);
      this.sdfVisPositions.set(state.sdfBodyPositions);

      // Transform gradient to each body's local frame (inline quatApplyInv to avoid per-call allocation)
      const sdfGradLocal = this._sdfGradLocal;
      for (let i = 0; i < this.numSdfBodies; i++) {
        const bw = state.sdfBodyQuaternions[i * 4];
        const bx = state.sdfBodyQuaternions[i * 4 + 1];
        const by = state.sdfBodyQuaternions[i * 4 + 2];
        const bz = state.sdfBodyQuaternions[i * 4 + 3];
        const gx = sdfGradWorld[i * 3];
        const gy = sdfGradWorld[i * 3 + 1];
        const gz = sdfGradWorld[i * 3 + 2];
        // quatApplyInv: q^{-1} * v * q (conjugate rotation)
        const tx = 2.0 * (by * gz - bz * gy);
        const ty = 2.0 * (bz * gx - bx * gz);
        const tz = 2.0 * (bx * gy - by * gx);
        const cx = by * tz - bz * ty;
        const cy = bz * tx - bx * tz;
        const cz = bx * ty - by * tx;
        sdfGradLocal[i * 3]     = gx - bw * tx + cx;
        sdfGradLocal[i * 3 + 1] = gy - bw * ty + cy;
        sdfGradLocal[i * 3 + 2] = gz - bw * tz + cz;
      }

      // interaction_sdf = concat(sdf_dist(7), sdf_grad_local(21)) = 28
      const interactionSdf = this._interactionSdf;
      interactionSdf.set(sdfDist, 0);
      interactionSdf.set(sdfGradLocal, this.numSdfBodies);

      /* ---- 2. Latent encoder ---- */
      // Reuse pre-created tensor (wraps _interactionSdf by reference, data already written above)
      const leOutput = await this.latentEncoderSession.run(this._leInputFeed);
      const rawLatent = leOutput[this.latentEncoderSession.outputNames[0]].data;

      // L2 normalise into pre-allocated buffer
      const latent = this._latent;
      let norm = 0;
      for (let i = 0; i < 64; i++) norm += rawLatent[i] * rawLatent[i];
      norm = Math.sqrt(norm) + 1e-8;
      for (let i = 0; i < 64; i++) latent[i] = rawLatent[i] / norm;

      // Dispose ORT output tensors to free WASM memory (input tensor is reused)
      for (const key in leOutput) { const t = leOutput[key]; if (t?.dispose) t.dispose(); }

      /* ---- 3. Update histories ---- */
      // joint_states = pos + vel*0.05
      const jointState = this._jointState;
      for (let i = 0; i < this.numActions; i++) {
        jointState[i] = state.jointPos[i];
        jointState[this.numActions + i] = state.jointVel[i] * 0.05;
      }
      // gravity in body frame: quatApplyInv(rootQuat, [0,0,-1]) — inlined to avoid allocation
      {
        const rw = state.rootQuat[0], rx = state.rootQuat[1], ry = state.rootQuat[2], rz = state.rootQuat[3];
        // v = [0, 0, -1]
        const tx = 2.0 * (ry * (-1) - rz * 0);   // 2*(y*vz - z*vy)
        const ty = 2.0 * (rz * 0 - rx * (-1));    // 2*(z*vx - x*vz)
        const tz = 2.0 * (rx * 0 - ry * 0);       // 2*(x*vy - y*vx)
        const cx = ry * tz - rz * ty;
        const cy = rz * tx - rx * tz;
        const cz = rx * ty - ry * tx;
        this._gravLocalBuf[0] = 0 - rw * tx + cx;
        this._gravLocalBuf[1] = 0 - rw * ty + cy;
        this._gravLocalBuf[2] = -1 - rw * tz + cz;
      }

      this.histJointStates.push(jointState);
      this.histGravities.push(this._gravLocalBuf);
      this.histRootAngVels.push(state.rootAngVel);
      this.histLastActions.push(this.lastActions);
      this.histInteractionLatents.push(latent);

      /* ---- 4. Build proprio (flatten histories into pre-allocated buffer) ---- */
      const proprio = this._proprio;
      let off = 0;
      const jsFlat = this.histJointStates.flatten();
      proprio.set(jsFlat, off); off += jsFlat.length;
      const gFlat = this.histGravities.flatten();
      proprio.set(gFlat, off); off += gFlat.length;
      const ravFlat = this.histRootAngVels.flatten();
      proprio.set(ravFlat, off); off += ravFlat.length;
      const laFlat = this.histLastActions.flatten();
      proprio.set(laFlat, off);

      /* ---- 5. Build movement_goal (all math inlined to avoid per-frame allocations) ---- */
      const movementGoal = this._movementGoal;
      {
        // yawComponent(rootQuat) → rootYawInv (conjugate of yaw-only quaternion)
        const rqw = state.rootQuat[0], rqx = state.rootQuat[1], rqy = state.rootQuat[2], rqz = state.rootQuat[3];
        const sinyCosp = 2.0 * (rqw * rqz + rqx * rqy);
        const cosyCosp = 1.0 - 2.0 * (rqy * rqy + rqz * rqz);
        const halfYaw = 0.5 * Math.atan2(sinyCosp, cosyCosp);
        let yw = Math.cos(halfYaw), yz = Math.sin(halfYaw);
        // normalize (yx=yy=0)
        const yn = Math.sqrt(yw * yw + yz * yz) || 1e-9;
        yw /= yn; yz /= yn;
        // rootYawInv = conjugate: [yw, 0, 0, -yz]
        const yiw = yw, yiz = -yz; // yix=yiy=0

        // _quatApplyForward(rootYawInv, dpos) — delta_pos_local
        const dpx = this.targetRootPos[0] - state.rootPos[0];
        const dpy = this.targetRootPos[1] - state.rootPos[1];
        // dpz = 0
        // q=[yiw,0,0,yiz], v=[dpx,dpy,0]: cross-product simplification
        const dp_tx = 2.0 * (0 * 0 - yiz * dpy);   // 2*(qy*vz - qz*vy)
        const dp_ty = 2.0 * (yiz * dpx - 0 * 0);   // 2*(qz*vx - qx*vz)
        const dp_tz = 2.0 * (0 * dpy - 0 * dpx);   // 2*(qx*vy - qy*vx)
        movementGoal[0] = dpx + yiw * dp_tx + (0 * dp_tz - yiz * dp_ty);
        movementGoal[1] = dpy + yiw * dp_ty + (yiz * dp_tx - 0 * dp_tz);
        movementGoal[2] = 0   + yiw * dp_tz + (0 * dp_ty - 0 * dp_tx);

        // quatMultiply(rootYawInv, targetRootQuat) → deltaQ
        const tqw = this.targetRootQuat[0], tqx = this.targetRootQuat[1];
        const tqy = this.targetRootQuat[2], tqz = this.targetRootQuat[3];
        // a=[yiw,0,0,yiz], b=[tqw,tqx,tqy,tqz]
        const dqw = yiw * tqw - yiz * tqz;                    // aw*bw - az*bz (ax=ay=0)
        const dqx = yiw * tqx + yiz * tqy;                    // aw*bx + az*by
        const dqy = yiw * tqy - yiz * tqx;                    // aw*by - az*bx
        const dqz = yiw * tqz + yiz * tqw;                    // aw*bz + az*bw

        // quatToMatrix(deltaQ) → first 2 columns of rotation matrix (6 values)
        const twoS = 2.0 / (dqw * dqw + dqx * dqx + dqy * dqy + dqz * dqz);
        // Row-major: r00,r01,r02, r10,r11,r12, r20,r21,r22
        // We only need first 2 cols: r00,r01, r10,r11, r20,r21
        movementGoal[3] = 1 - twoS * (dqy * dqy + dqz * dqz);  // r00
        movementGoal[4] = twoS * (dqx * dqy - dqz * dqw);      // r01
        movementGoal[5] = twoS * (dqx * dqy + dqz * dqw);      // r10
        movementGoal[6] = 1 - twoS * (dqx * dqx + dqz * dqz);  // r11
        movementGoal[7] = twoS * (dqx * dqz - dqy * dqw);      // r20
        movementGoal[8] = twoS * (dqy * dqz + dqx * dqw);      // r21

        // _quatApplyForward(rootYawInv, targetVel) — target_root_vel_local
        const tvx = this.targetRootPos[0] - this.prevTargetRootPos[0];
        const tvy = this.targetRootPos[1] - this.prevTargetRootPos[1];
        // tvz = 0
        const tv_tx = 2.0 * (0 * 0 - yiz * tvy);
        const tv_ty = 2.0 * (yiz * tvx - 0 * 0);
        const tv_tz = 0.0;
        movementGoal[9]  = tvx + yiw * tv_tx + (0 * tv_tz - yiz * tv_ty);
        movementGoal[10] = tvy + yiw * tv_ty + (yiz * tv_tx - 0 * tv_tz);
        movementGoal[11] = 0   + yiw * tv_tz + (0 * tv_ty - 0 * tv_tx);
      }

      // Store for next velocity computation
      this.prevTargetRootPos.set(this.targetRootPos);

      /* ---- 6. interaction_field (latent history) ---- */
      // flatten() populates histInteractionLatents._flat buffer in-place;
      // the _policyTensors['interaction_field'] tensor wraps that same buffer.
      this.histInteractionLatents.flatten();

      /* ---- 7. Run policy ONNX ---- */
      // Reuse pre-created tensors (wrap pre-allocated arrays by reference, data already written)
      // histInteractionLatents.flatten() writes into histInteractionLatents._flat which
      // is the same buffer the _policyTensors['interaction_field'] tensor wraps.
      const policyOutput = await this.policySession.run(this._policyFeed);

      // Find the action output
      let actionData = null;
      for (const outName of this.policySession.outputNames) {
        if (outName === 'action' || outName.includes('action')) {
          actionData = policyOutput[outName].data;
          break;
        }
      }
      if (!actionData) {
        actionData = policyOutput[this.policySession.outputNames[0]].data;
      }

      /* ---- 8. Process actions (reuse pre-allocated buffer) ---- */
      const clip = this.actionClip;
      const target = this._target;
      for (let i = 0; i < this.numActions; i++) {
        const v = Math.max(-clip, Math.min(clip, actionData[i]));
        this.lastActions[i] = v;
        target[i] = v;
      }

      // Dispose ORT output tensors to free WASM memory (input tensors are reused)
      for (const key in policyOutput) { const t = policyOutput[key]; if (t?.dispose) t.dispose(); }

      return target;
    } finally {
      this.isInferencing = false;
    }
  }
}
