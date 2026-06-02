import * as ort from 'onnxruntime-web';
import { ONNXModule } from './onnxHelper.js';
import { Observations } from './observationHelpers.js';
import { TrackingHelper } from './trackingHelper.js';
import { toFloatArray } from './utils/math.js';

export class PolicyRunner {
  constructor(config, options = {}) {
    this.config = config;
    this.policyJointNames = (options.policyJointNames ?? config.policy_joint_names ?? []).slice();
    if (this.policyJointNames.length === 0) {
      throw new Error('PolicyRunner requires policy_joint_names in config');
    }
    this.numActions = this.policyJointNames.length;

    this.actionScale = toFloatArray(options.actionScale ?? config.action_scale, this.numActions, 1.0);
    this.defaultJointPos = toFloatArray(options.defaultJointPos ?? [], this.numActions, 0.0);
    this.actionClip = typeof config.action_clip === 'number' ? config.action_clip : 10.0;

    this.module = new ONNXModule(config.onnx);
    this.inputDict = {};
    this.isInferencing = false;
    this.lastActions = new Float32Array(this.numActions);

    this.tracking = null;
    if (config.tracking) {
      this.tracking = new TrackingHelper({
        ...config.tracking,
        policy_joint_names: this.policyJointNames
      });
    }

    this.obsModules = this._buildObsModules(config.obs_config);
    this.numObs = this.obsModules.reduce((sum, obs) => sum + (obs.size ?? 0), 0);
  }

  async init() {
    await this.module.init();
    this.reset();
    // Pre-allocate reusable buffers
    this._obsBuffer = new Float32Array(this.numObs);
    this._targetBuffer = new Float32Array(this.numActions);
    // Create reusable policy input tensor (wraps _obsBuffer by reference)
    this._policyTensor = new ort.Tensor('float32', this._obsBuffer, [1, this._obsBuffer.length]);
  }

  async dispose() {
    // Dispose the reusable policy tensor
    if (this._policyTensor) {
      try { this._policyTensor.dispose(); } catch (e) { /* ignore */ }
      this._policyTensor = null;
    }
    // Dispose any ORT tensors still in inputDict (carry tensors)
    for (const v of Object.values(this.inputDict)) {
      if (v?.dispose) try { v.dispose(); } catch (e) { /* ignore */ }
    }
    this.inputDict = {};
    if (this.module?.dispose) {
      await this.module.dispose();
    }
  }

  _buildObsModules(obsConfig) {
    const obsList = (obsConfig && Array.isArray(obsConfig.policy)) ? obsConfig.policy : [];
    return obsList.map((obsConfigEntry) => {
      const ObsClass = Observations[obsConfigEntry.name];
      if (!ObsClass) {
        throw new Error(`Unknown observation type: ${obsConfigEntry.name}`);
      }
      const kwargs = { ...obsConfigEntry };
      delete kwargs.name;
      return new ObsClass(this, kwargs);
    });
  }

  reset(state = null) {
    // Dispose old ORT tensors before replacing, but skip reusable tensors
    // owned by ONNXModule (is_init) and our reusable policy tensor
    for (const [k, v] of Object.entries(this.inputDict)) {
      if (v?.dispose && v !== this._policyTensor && v !== this.module?._isInitTrue && v !== this.module?._isInitFalse) {
        try { v.dispose(); } catch (e) { /* ignore */ }
      }
    }
    this.inputDict = this.module.initInput() ?? {};
    this.lastActions.fill(0.0);
    if (this.tracking) {
      this.tracking.reset(state);
    }
    for (const obs of this.obsModules) {
      if (typeof obs.reset === 'function') {
        obs.reset(state);
      }
    }
  }

  async step(state) {
    if (this.isInferencing) {
      return null;
    }

    if (!state) {
      throw new Error('PolicyRunner.step requires a state object');
    }

    this.isInferencing = true;
    try {
      if (this.tracking) {
        this.tracking.advance();
      }

      const obsForPolicy = this._obsBuffer;
      let offset = 0;
      for (const obs of this.obsModules) {
        if (typeof obs.update === 'function') {
          obs.update(state);
        }
        const obsValue = obs.compute(state);
        const obsArray = ArrayBuffer.isView(obsValue) ? obsValue : Float32Array.from(obsValue);
        obsForPolicy.set(obsArray, offset);
        offset += obsArray.length;
      }

      // Reuse pre-created policy tensor (wraps _obsBuffer by reference, data already written above)
      this.inputDict['policy'] = this._policyTensor;

      const [result, carry] = await this.module.runInference(this.inputDict);

      // Collect carry tensor values so we don't dispose them below
      const carryValues = new Set(Object.values(carry));

      // Dispose old carry tensors from inputDict before overwriting,
      // but skip the reusable is_init tensor (managed by ONNXModule)
      for (const key of Object.keys(carry)) {
        const old = this.inputDict[key];
        if (old?.dispose && !carryValues.has(old) && old !== this.module._isInitTrue && old !== this.module._isInitFalse) {
          old.dispose();
        }
      }
      Object.assign(this.inputDict, carry);

      const action = result['action']?.data;
      if (!action || action.length !== this.numActions) {
        throw new Error('PolicyRunner received invalid action output');
      }

      const clip = typeof this.actionClip === 'number' ? this.actionClip : Infinity;
      for (let i = 0; i < this.numActions; i++) {
        const value = action[i];
        const clamped = clip !== Infinity ? Math.max(-clip, Math.min(clip, value)) : value;
        this.lastActions[i] = clamped;
      }

      const target = this._targetBuffer;
      for (let i = 0; i < this.numActions; i++) {
        target[i] = this.defaultJointPos[i] + this.actionScale[i] * this.lastActions[i];
      }

      // Dispose ORT output tensors, but skip any that were moved to carry
      // (e.g., adapt_hx is stored in carry and will be used as input next frame)
      for (const v of Object.values(result)) {
        if (v?.dispose && !carryValues.has(v)) v.dispose();
      }

      return target;
    } finally {
      this.isInferencing = false;
    }
  }
}
