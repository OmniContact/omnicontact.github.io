import * as ort from 'onnxruntime-web';

// Configure ORT WASM: single-thread avoids SharedArrayBuffer memory growth
ort.env.wasm.numThreads = 1;

export class ONNXModule {
  constructor(config) {
    this.modelPath = config.path;
    this.metaData = config.meta;
    this.isRecurrent = config.meta.in_keys.includes("adapt_hx");
    console.log("isRecurrent", this.isRecurrent);
  }

  async init() {
    // Load the ONNX model
    const modelResponse = await fetch(this.modelPath);
    const modelArrayBuffer = await modelResponse.arrayBuffer();

    this.inKeys = this.metaData["in_keys"];
    this.outKeys = this.metaData["out_keys"];

    // Create session from the array buffer
    this.session = await ort.InferenceSession.create(modelArrayBuffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      enableMemPattern: true,
      enableCpuMemArena: true,
    });

    console.log('ONNX model loaded successfully');
    console.log("inKeys", this.inKeys);
    console.log("outKeys", this.outKeys);

    console.log("inputNames", this.session.inputNames);
    console.log("outputNames", this.session.outputNames);
  }

  initInput() {
    if (this.isRecurrent) {
      // Create reusable is_init tensors (JS-backed, no WASM allocation)
      this._isInitTrue = new ort.Tensor('bool', [true], [1]);
      this._isInitFalse = new ort.Tensor('bool', [false], [1]);
      return {
        "is_init": this._isInitTrue,
        "adapt_hx": new ort.Tensor('float32', new Float32Array(128), [1, 128])
      }
    } else {
      this._isInitTrue = null;
      this._isInitFalse = null;
      return {};
    }
  }

  async runInference(input) {
    // construct input
    let onnxInput = {};
    for (let i = 0; i < this.inKeys.length; i++) {
      onnxInput[this.session.inputNames[i]] = input[this.inKeys[i]];
    }
    // run inference
    const onnxOutput = await this.session.run(onnxInput);
    // construct output
    let result = {};
    for (let i = 0; i < this.outKeys.length; i++) {
      result[this.outKeys[i]] = onnxOutput[this.session.outputNames[i]];
    }
    // carry contains tensors that persist to the next inference step
    // For recurrent models: is_init becomes false, adapt_hx carries forward
    // IMPORTANT: carry["adapt_hx"] is the SAME object as result["next,adapt_hx"].
    // The caller must NOT dispose carry values when disposing result.
    let carry = {};
    if (this.isRecurrent) {
      carry["is_init"] = this._isInitFalse; // reuse pre-created tensor
      carry["adapt_hx"] = result["next,adapt_hx"];
    }
    return [result, carry];
  }

  async dispose() {
    if (this.session) {
      try { await this.session.release(); } catch (e) { /* ignore */ }
      this.session = null;
    }
  }
}
