import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { DragStateManager } from './utils/DragStateManager.js';
import { downloadExampleScenesFolder, getPosition, getQuaternion, getMujocoQuaternion, toMujocoPos, reloadScene, reloadPolicy, reloadInteractionPolicy } from './mujocoUtils.js';
import { parseObj, parseStl, transformObjToMujoco, computeSdfGrid, buildThreeGeometry } from './objSdfComputer.js';
import { SDFVoxelGrid } from './sdfHelper.js';
import { eulerToQuat, normalizeQuat } from './utils/math.js';

const queryParams = new URLSearchParams(window.location.search);
const defaultPolicy = queryParams.get('force_policy')
  ?? queryParams.get('policy')
  ?? "./examples/checkpoints/g1/carrybox_wtac_policy.json";
const DEFAULT_CARRYBOX_SCENE = 'g1/carrybox_manager_carry.xml';
const LOCAL_PUSHBOX_SCENE = 'g1/omniplan2track_push_box.xml';

// Pre-allocated for createThickArrow.setDirection
const _thickArrowUp = new THREE.Vector3(0, 1, 0);
const _thickArrowAxis = new THREE.Vector3();

/**
 * Create a mesh-based arrow (cylinder shaft + cone head) for cross-platform thick rendering.
 * WebGL's linewidth is capped at 1px on most platforms, so we use geometry instead.
 * @param {number} color - hex color
 * @param {number} shaftRadius - cylinder radius for the shaft
 * @param {number} opacity
 * @returns {THREE.Group} with setDirection(dir) and setLength(len, headLen, headWidth) methods
 */
function createThickArrow(color, shaftRadius = 0.01, opacity = 0.8) {
  const group = new THREE.Group();
  const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthTest: true });

  // Shaft: unit cylinder along +Y, base at origin
  const shaftGeo = new THREE.CylinderGeometry(shaftRadius, shaftRadius, 1, 8);
  shaftGeo.translate(0, 0.5, 0);
  const shaft = new THREE.Mesh(shaftGeo, mat);
  group.add(shaft);

  // Head: unit cone along +Y, base at origin
  const headGeo = new THREE.ConeGeometry(0.5, 1, 8);
  headGeo.translate(0, 0.5, 0);
  const head = new THREE.Mesh(headGeo, mat.clone());
  group.add(head);

  group._shaft = shaft;
  group._head = head;

  group.setDirection = function(dir) {
    if (dir.lengthSq() < 1e-10) return;
    const d = dir.lengthSq() !== 1 ? dir.clone().normalize() : dir;
    if (Math.abs(d.y) > 0.99999) {
      this.quaternion.set(0, 0, 0, d.y > 0 ? 1 : 0);
      if (d.y < 0) this.quaternion.set(1, 0, 0, 0);
    } else {
      _thickArrowAxis.crossVectors(_thickArrowUp, d).normalize();
      const angle = Math.acos(Math.max(-1, Math.min(1, d.y)));
      this.quaternion.setFromAxisAngle(_thickArrowAxis, angle);
    }
  };

  group.setLength = function(length, headLength, headWidth) {
    if (headLength === undefined) headLength = 0.2 * length;
    if (headWidth === undefined) headWidth = 0.4 * headLength;
    const shaftLen = Math.max(0.0001, length - headLength);
    this._shaft.scale.set(1, shaftLen, 1);
    this._head.scale.set(headWidth * 2, headLength, headWidth * 2);
    this._head.position.set(0, shaftLen, 0);
  };

  return group;
}

/**
 * Convert parsed OBJ vertex/face data to binary STL with THREE.js→MuJoCo coordinate transform.
 * Skips degenerate faces (zero area) to produce a clean mesh for MuJoCo convex hull.
 */
function objToStlBinary(vertices, faces, numFaces) {
  // First pass: count valid (non-degenerate) faces
  // Vertices are already in MuJoCo space (Z-up) after transformObjToMujoco(), no swizzle needed
  const MIN_AREA_SQ = 1e-20;
  let validCount = 0;
  for (let f = 0; f < numFaces; f++) {
    const i0 = faces[f * 3] * 3, i1 = faces[f * 3 + 1] * 3, i2 = faces[f * 3 + 2] * 3;
    // Check for out-of-bounds indices
    if (i0 + 2 >= vertices.length || i1 + 2 >= vertices.length || i2 + 2 >= vertices.length) continue;
    const v0x = vertices[i0], v0y = vertices[i0 + 1], v0z = vertices[i0 + 2];
    const v1x = vertices[i1], v1y = vertices[i1 + 1], v1z = vertices[i1 + 2];
    const v2x = vertices[i2], v2y = vertices[i2 + 1], v2z = vertices[i2 + 2];
    const e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    const e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    if (nx * nx + ny * ny + nz * nz > MIN_AREA_SQ) validCount++;
  }
  if (validCount === 0) validCount = 1; // Ensure at least a minimal buffer

  const buf = new ArrayBuffer(80 + 4 + validCount * 50);
  const view = new DataView(buf);
  view.setUint32(80, validCount, true);
  let off = 84;
  let written = 0;
  for (let f = 0; f < numFaces; f++) {
    const i0 = faces[f * 3] * 3, i1 = faces[f * 3 + 1] * 3, i2 = faces[f * 3 + 2] * 3;
    if (i0 + 2 >= vertices.length || i1 + 2 >= vertices.length || i2 + 2 >= vertices.length) continue;
    // Vertices already in MuJoCo space, pass through directly
    const v0x = vertices[i0], v0y = vertices[i0 + 1], v0z = vertices[i0 + 2];
    const v1x = vertices[i1], v1y = vertices[i1 + 1], v1z = vertices[i1 + 2];
    const v2x = vertices[i2], v2y = vertices[i2 + 1], v2z = vertices[i2 + 2];
    const e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    const e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;
    let nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    const lenSq = nx * nx + ny * ny + nz * nz;
    if (lenSq <= MIN_AREA_SQ) continue; // Skip degenerate face
    const len = Math.sqrt(lenSq);
    nx /= len; ny /= len; nz /= len;
    view.setFloat32(off, nx, true); view.setFloat32(off + 4, ny, true); view.setFloat32(off + 8, nz, true); off += 12;
    view.setFloat32(off, v0x, true); view.setFloat32(off + 4, v0y, true); view.setFloat32(off + 8, v0z, true); off += 12;
    view.setFloat32(off, v1x, true); view.setFloat32(off + 4, v1y, true); view.setFloat32(off + 8, v1z, true); off += 12;
    view.setFloat32(off, v2x, true); view.setFloat32(off + 4, v2y, true); view.setFloat32(off + 8, v2z, true); off += 12;
    view.setUint16(off, 0, true); off += 2;
    written++;
  }
  // Update actual face count (may differ from initial estimate if some were skipped)
  view.setUint32(80, written, true);
  return new Uint8Array(buf, 0, 84 + written * 50);
}

function normalizePlannerTask(task) {
  const key = String(task ?? 'carrybox').trim().toLowerCase().replace(/_/g, '-');
  if (key === 'push' || key === 'pushbox') {
    return 'pushbox';
  }
  if (key === 'push-old' || key === 'pushbox-old' || key === 'old-push') {
    return 'pushbox-old';
  }
  return 'carrybox';
}

function isPushPlannerTask(task) {
  const normalized = normalizePlannerTask(task);
  return normalized === 'pushbox' || normalized === 'pushbox-old';
}

function isPush2PlannerTask(task) {
  return normalizePlannerTask(task) === 'pushbox-old';
}

function plannerObjectBodyName(task) {
  return 'box';
}

function plannerGhostBodyName(task) {
  if (isPushPlannerTask(task)) {
    return null;
  }
  return 'ghost_box';
}

function plannerBoxHalfDims(task, carryDims = [0.15, 0.15, 0.15], pushDims = [0.23, 0.25, 0.26]) {
  return Array.from(isPushPlannerTask(task) ? pushDims : carryDims);
}

function plannerSceneForTask(task) {
  return isPushPlannerTask(task) ? LOCAL_PUSHBOX_SCENE : DEFAULT_CARRYBOX_SCENE;
}

function enforcePlannerObjectHeights(task, boxStartPos, goalPos, halfDims) {
  const start = boxStartPos ? Array.from(boxStartPos) : null;
  const goal = goalPos ? Array.from(goalPos) : null;
  if (isPushPlannerTask(task)) {
    const halfHeight = Number(halfDims?.[2] ?? 0.26);
    if (start) start[2] = halfHeight;
    if (goal) goal[2] = halfHeight;
  }
  return { boxStartPos: start, goalPos: goal };
}

export class MuJoCoDemo {
  constructor(mujoco) {
    this.mujoco = mujoco;
    mujoco.FS.mkdir('/working');
    mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');

    this.params = {
      paused: true,
      current_motion: 'default'
    };
    this.policyRunner = null;
    this.interactionRunner = null;   // InteractionPolicyRunner
    this.isInteractionMode = false;
    this.kpPolicy = null;
    this.kdPolicy = null;
    this.actionTarget = null;
    this.model = null;
    this.data = null;
    this.simulation = null;
    this.currentPolicyPath = defaultPolicy;
    this.carryBoxPlannerConfig = null;

    // Body name → body ID lookup (populated on scene load)
    this.bodyNameToId = {};
    this.geomNameToId = {};
    this.geomIdToMesh = {};
    // SDF query body IDs (populated on interaction policy load)
    this.sdfBodyIds = [];
    // Scene object body IDs
    this.sceneObjectBodyIds = {};

    this.bodies = {};
    this.lights = {};

    this.container = document.getElementById('mujoco-container');

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.001, 100);
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set(3.0, 2.2, 3.0);
    this.scene.add(this.camera);

    this.scene.background = new THREE.Color(0.22, 0.32, 0.42);
    this.scene.fog = null;

    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.35);
    this.ambientLight.name = 'AmbientLight';
    this.scene.add(this.ambientLight);

    // Detect Safari for reduced GPU settings (Safari's JSC GC handles WASM-backed
    // TypedArray views poorly, and large framebuffers exacerbate memory pressure)
    const ua = navigator.userAgent;
    this._isSafari = /Safari\//.test(ua)
      && !/Chrome\//.test(ua) && !/Chromium\//.test(ua)
      && !/Edg\//.test(ua) && !/OPR\//.test(ua);

    this.renderer = new THREE.WebGLRenderer({ antialias: !this._isSafari });
    this.renderScale = this._isSafari ? 1.0 : 2.0;
    this.renderer.setPixelRatio(this.renderScale);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = !this._isSafari;
    if (!this._isSafari) {
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    }

    this.simStepHz = 0;
    this._stepFrameCount = 0;
    this._stepLastTime = performance.now();
    this._lastRenderTime = 0;

    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.7, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.10;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    this._resizeHandler = this.onWindowResize.bind(this);
    window.addEventListener('resize', this._resizeHandler);

    this.dragStateManager = new DragStateManager(this.scene, this.renderer, this.camera, this.container.parentElement, this.controls);

    // Root command visualization arrows (mesh-based for visible thickness)
    this.rootCmdArrow = createThickArrow(0x00cc66, 0.005, 0.7);
    this.rootCmdArrow.visible = false;
    this.scene.add(this.rootCmdArrow);

    this.rootCmdHeading = createThickArrow(0x00bbff, 0.005, 0.7);
    this.rootCmdHeading.visible = false;
    this.scene.add(this.rootCmdHeading);

    // SDF visualization arrows (one per query body, created lazily)
    this.showSdfVis = true;
    this.autoReplanEnabled = false;
    this.autoReplanState = {
      hasHeldBox: false,
      detachCounter: 0,
      refMismatchCounter: 0,
      cooldownCounter: 0,
      pendingReplan: false,
      pendingReason: '',
      staticCounter: 0,
      replanCounter: 0,
      lastReason: ''
    };
    this.autoReplanConfig = {
      detachDistThreshold: 0.32,
      detachTriggerTicks: 8,
      replanCooldownTicks: 80,
      goalTolerance: 0.2,
      refMismatchThreshold: 0.18,
      refMismatchTriggerTicks: 8,
      objSpeedThreshold: 0.05,
      objAngSpeedThreshold: 0.15,
      staticTicks: 5,
      uprightZThreshold: 0.92,
      liftedHeightThreshold: 0.12
    };
    this._sdfArrows = [];
    this._tmpSdfOrigin = new THREE.Vector3();
    this._tmpSdfDir = new THREE.Vector3();

    // Pre-allocated temporaries for render() to avoid per-frame GC pressure
    this._tmpTargetThree = new THREE.Vector3();
    this._tmpRobotPos = new THREE.Vector3();
    this._tmpDir = new THREE.Vector3();
    this._tmpHeadingDir = new THREE.Vector3();
    this._tmpLightTarget = new THREE.Vector3();

    this.followEnabled = false;
    this.followHeight = 0.75;
    this.followLerp = 0.05;
    this.followTarget = new THREE.Vector3();
    this.followTargetDesired = new THREE.Vector3();
    this.followDelta = new THREE.Vector3();
    this.followOffset = new THREE.Vector3();
    this.followInitialized = false;
    this.followBodyId = null;
    this.followDistance = this.camera.position.distanceTo(this.controls.target);

    this.lastSimState = {
      bodies: new Map(),
      lights: new Map(),
      tendons: {
        numWraps: 0,
        matrix: new THREE.Matrix4()
      }
    };

    this.renderer.setAnimationLoop(this.render.bind(this));

    this.reloadScene = reloadScene.bind(this);
    this.reloadPolicy = reloadPolicy.bind(this);
    this.reloadInteractionPolicy = reloadInteractionPolicy.bind(this);

    // Memory monitoring
    this._memMonitorInterval = null;
    this._memBaseline = null;
  }

  /** Start periodic memory monitoring (logs every 10s). */
  startMemoryMonitor() {
    if (this._memMonitorInterval) return;
    this._memBaseline = null;
    this._memMonitorInterval = setInterval(() => {
      const mem = performance.memory; // Chrome only
      if (mem) {
        const usedMB = (mem.usedJSHeapSize / 1048576).toFixed(1);
        const totalMB = (mem.totalJSHeapSize / 1048576).toFixed(1);
        const limitMB = (mem.jsHeapSizeLimit / 1048576).toFixed(1);
        if (!this._memBaseline) this._memBaseline = mem.usedJSHeapSize;
        const deltaMB = ((mem.usedJSHeapSize - this._memBaseline) / 1048576).toFixed(1);
        console.log(`[MEM] JS Heap: ${usedMB}/${totalMB} MB (limit ${limitMB}), delta: +${deltaMB} MB`);
      }
      // Log THREE.js render info
      const info = this.renderer.info;
      console.log(`[MEM] THREE: geometries=${info.memory.geometries}, textures=${info.memory.textures}, programs=${info.programs?.length ?? '?'}, drawCalls=${info.render.calls}`);
    }, 10000);
  }

  stopMemoryMonitor() {
    if (this._memMonitorInterval) {
      clearInterval(this._memMonitorInterval);
      this._memMonitorInterval = null;
    }
  }

  async init() {
    await downloadExampleScenesFolder(this.mujoco);

    // Check if defaultPolicy is interaction type
    const configResp = await fetch(defaultPolicy);
    const configJson = await configResp.json();
    const sceneXml = configJson.scene_xml ?? 'g1/g1.xml';
    const isInteraction = configJson.policy_type === 'interaction';

    this._currentSceneFile = sceneXml;
    await this.reloadScene(sceneXml);
    this.updateFollowBodyId();

    if (isInteraction) {
      await this.reloadInteractionPolicy(defaultPolicy, configJson);
    } else {
      await this.reloadPolicy(defaultPolicy);
    }
    this.alive = true;
  }

  async reload(mjcf_path) {
    await this.reloadScene(mjcf_path);
    this._currentSceneFile = mjcf_path;
    this.updateFollowBodyId();
    this.timestep = this.model.opt.timestep;
    this.decimation = Math.max(1, Math.round(0.02 / this.timestep));

    console.log('timestep:', this.timestep, 'decimation:', this.decimation);

    await this.reloadPolicy(this.currentPolicyPath ?? defaultPolicy);
    this.alive = true;
  }

  async _ensurePlannerSceneForTask(task) {
    const sceneFile = plannerSceneForTask(task);
    if (this._currentSceneFile === sceneFile) {
      return false;
    }

    const wasPaused = this.params.paused;
    this.params.paused = true;
    this.actionTarget = null;
    await this.reloadScene(sceneFile);
    this._currentSceneFile = sceneFile;
    this.updateFollowBodyId();
    await this.reloadPolicy(this.currentPolicyPath ?? defaultPolicy);
    this.params.paused = wasPaused;
    return true;
  }

  setFollowEnabled(enabled) {
    this.followEnabled = Boolean(enabled);
    this.followInitialized = false;
    if (this.followEnabled) {
      this.followOffset.subVectors(this.camera.position, this.controls.target);
      if (this.followOffset.lengthSq() === 0) {
        this.followOffset.set(0, 0, 1);
      }
      this.followOffset.setLength(this.followDistance);
      this.camera.position.copy(this.controls.target).add(this.followOffset);
      this.controls.update();
    }
  }

  setSdfVisEnabled(enabled) {
    this.showSdfVis = Boolean(enabled);
    // Hide all arrows when disabled
    if (!this.showSdfVis) {
      for (const arrow of this._sdfArrows) {
        arrow.visible = false;
      }
    }
  }

  /** Lazily create or retrieve SDF visualization arrows. */
  _ensureSdfArrows(count) {
    while (this._sdfArrows.length < count) {
      const arrow = createThickArrow(0xff4444, 0.005, 0.8);
      arrow.visible = false;
      this.scene.add(arrow);
      this._sdfArrows.push(arrow);
    }
  }

  setAutoReplanEnabled(enabled) {
    this.autoReplanEnabled = Boolean(enabled);
    this.resetAutoReplanState();
    return this.autoReplanEnabled;
  }

  getAutoReplanEnabled() {
    return this.autoReplanEnabled;
  }

  resetAutoReplanState() {
    this.autoReplanState.hasHeldBox = false;
    this.autoReplanState.detachCounter = 0;
    this.autoReplanState.refMismatchCounter = 0;
    this.autoReplanState.cooldownCounter = 0;
    this.autoReplanState.pendingReplan = false;
    this.autoReplanState.pendingReason = '';
    this.autoReplanState.staticCounter = 0;
    this.autoReplanState.lastReason = '';
  }

  _bodyMjPos(bodyName) {
    const bodyId = this.bodyNameToId?.[bodyName];
    if (bodyId === undefined || !this.simulation?.xpos) {
      return null;
    }
    const b = bodyId * 3;
    return [
      this.simulation.xpos[b],
      this.simulation.xpos[b + 1],
      this.simulation.xpos[b + 2]
    ];
  }

  _bodyMjQuat(bodyName) {
    const bodyId = this.bodyNameToId?.[bodyName];
    if (bodyId === undefined || !this.simulation?.xquat) {
      return null;
    }
    const b = bodyId * 4;
    return normalizeQuat([
      this.simulation.xquat[b],
      this.simulation.xquat[b + 1],
      this.simulation.xquat[b + 2],
      this.simulation.xquat[b + 3]
    ]);
  }

  _freeJointAdrForBody(bodyName) {
    const bodyId = this.bodyNameToId?.[bodyName];
    if (bodyId === undefined || !this.model) {
      return null;
    }
    const jointAdr = this.model.body_jntadr[bodyId];
    const jointNum = this.model.body_jntnum[bodyId];
    if (jointNum <= 0 || jointAdr < 0) {
      return null;
    }
    return {
      qpos: this.model.jnt_qposadr[jointAdr],
      qvel: this.model.jnt_dofadr[jointAdr]
    };
  }

  _boxPalmMinDist() {
    const boxPos = this._bodyMjPos(this._activePlannerObjectBodyName());
    if (!boxPos) {
      return Infinity;
    }
    const dists = [];
    for (const name of ['left_palm_link', 'right_palm_link']) {
      const pos = this._bodyMjPos(name);
      if (pos) {
        dists.push(Math.hypot(boxPos[0] - pos[0], boxPos[1] - pos[1], boxPos[2] - pos[2]));
      }
    }
    return dists.length ? Math.min(...dists) : Infinity;
  }

  _isBoxHeldForReplan() {
    const minDist = this._boxPalmMinDist();
    const boxPos = this._bodyMjPos(this._activePlannerObjectBodyName());
    const groundCenterZ = this.policyRunner?.boxHalfDims?.[2] ?? 0.15;
    const lifted = boxPos
      ? boxPos[2] >= groundCenterZ + this.autoReplanConfig.liftedHeightThreshold
      : false;
    return {
      held: minDist <= this.autoReplanConfig.detachDistThreshold && lifted,
      minDist,
      inContact: false,
      lifted
    };
  }

  _boxSpeedForReplan() {
    const adr = this._freeJointAdrForBody(this._activePlannerObjectBodyName());
    if (!adr || adr.qvel < 0 || !this.simulation?.qvel) {
      return { linear: 0, angular: 0 };
    }
    const qvel = this.simulation.qvel;
    return {
      linear: Math.hypot(qvel[adr.qvel], qvel[adr.qvel + 1], qvel[adr.qvel + 2]),
      angular: Math.hypot(qvel[adr.qvel + 3], qvel[adr.qvel + 4], qvel[adr.qvel + 5])
    };
  }

  _boxNearGoalForReplan() {
    const boxPos = this._bodyMjPos(this._activePlannerObjectBodyName());
    const goal = this.carryBoxPlannerConfig?.goalPos ?? this.policyRunner?.goalPos;
    if (!boxPos || !goal) {
      return false;
    }
    return Math.hypot(boxPos[0] - goal[0], boxPos[1] - goal[1], boxPos[2] - goal[2]) <= this.autoReplanConfig.goalTolerance;
  }

  _boxReferenceErrorForReplan() {
    const boxPos = this._bodyMjPos(this._activePlannerObjectBodyName());
    const refs = this.policyRunner?.refObjectPos;
    if (!boxPos || !refs?.length) {
      return 0;
    }
    const idx = Math.min(this.policyRunner.counterStep ?? 0, refs.length - 1);
    const ref = refs[idx];
    return Math.hypot(boxPos[0] - ref[0], boxPos[1] - ref[1], boxPos[2] - ref[2]);
  }

  _quatLocalZWorld(quat) {
    const [w, x, y, z] = normalizeQuat(quat);
    return [
      2.0 * (x * z + w * y),
      2.0 * (y * z - w * x),
      1.0 - 2.0 * (x * x + y * y)
    ];
  }

  _uprightYawQuatFrom(quat) {
    const [w, x, y, z] = normalizeQuat(quat);
    const xAxisX = 1.0 - 2.0 * (y * y + z * z);
    const xAxisY = 2.0 * (x * y + w * z);
    const yaw = Math.hypot(xAxisX, xAxisY) < 1e-6 ? 0.0 : Math.atan2(xAxisY, xAxisX);
    return normalizeQuat([Math.cos(0.5 * yaw), 0.0, 0.0, Math.sin(0.5 * yaw)]);
  }

  _boxHalfDimsForTask(task) {
    return plannerBoxHalfDims(
      task,
      this.policyRunner?.carryBoxHalfDims ?? [0.15, 0.15, 0.15],
      this.policyRunner?.pushBoxHalfDims ?? [0.23, 0.25, 0.26]
    );
  }

  _activePlannerTask() {
    return normalizePlannerTask(this.carryBoxPlannerConfig?.task ?? this.policyRunner?.task ?? 'carrybox');
  }

  _activePlannerObjectBodyName(task = null) {
    return plannerObjectBodyName(task ?? this._activePlannerTask());
  }

  _push2LineConstraint() {
    const task = this._activePlannerTask();
    if (!isPush2PlannerTask(task)) {
      return null;
    }
    const origin = this.carryBoxPlannerConfig?.boxStartPos
      ?? this.policyRunner?.refObjectPos?.[0]
      ?? null;
    const goal = this.carryBoxPlannerConfig?.goalPos
      ?? this.policyRunner?.goalPos
      ?? null;
    if (!origin || !goal) {
      return null;
    }
    const dx = Number(goal[0]) - Number(origin[0]);
    const dy = Number(goal[1]) - Number(origin[1]);
    const len = Math.hypot(dx, dy);
    if (len < 1e-6) {
      return null;
    }
    return {
      origin: [Number(origin[0]), Number(origin[1])],
      axis: [dx / len, dy / len]
    };
  }

  _constrainPush2BoxMotion() {
    const line = this._push2LineConstraint();
    const adr = line ? this._freeJointAdrForBody(this._activePlannerObjectBodyName('pushbox-old')) : null;
    if (!line || !adr || adr.qpos < 0 || !this.simulation?.qpos) {
      return;
    }

    const qpos = this.simulation.qpos;
    const relX = qpos[adr.qpos] - line.origin[0];
    const relY = qpos[adr.qpos + 1] - line.origin[1];
    const along = relX * line.axis[0] + relY * line.axis[1];
    qpos[adr.qpos] = line.origin[0] + along * line.axis[0];
    qpos[adr.qpos + 1] = line.origin[1] + along * line.axis[1];
    qpos.set([1, 0, 0, 0], adr.qpos + 3);

    const qvel = this.simulation.qvel;
    if (qvel && adr.qvel >= 0) {
      const vx = qvel[adr.qvel];
      const vy = qvel[adr.qvel + 1];
      const vAlong = vx * line.axis[0] + vy * line.axis[1];
      qvel[adr.qvel] = vAlong * line.axis[0];
      qvel[adr.qvel + 1] = vAlong * line.axis[1];
      qvel[adr.qvel + 3] = 0;
      qvel[adr.qvel + 4] = 0;
      qvel[adr.qvel + 5] = 0;
    }
    this.simulation.forward();
  }

  _setFreeJointBodyPose(bodyName, pos, quat = [1, 0, 0, 0]) {
    const adr = this._freeJointAdrForBody(bodyName);
    if (!adr || adr.qpos < 0 || !this.simulation?.qpos) {
      return;
    }
    this.simulation.qpos.set(pos, adr.qpos);
    this.simulation.qpos.set(quat, adr.qpos + 3);
    if (this.simulation.qvel && adr.qvel >= 0) {
      this.simulation.qvel.fill(0, adr.qvel, adr.qvel + 6);
    }
  }

  _moveInactivePlannerBoxesAway(task) {
    const activeObject = plannerObjectBodyName(task);
    const activeGhost = plannerGhostBodyName(task);
    const farPos = [100, 100, 2.0];
    for (const bodyName of ['box', 'box_push', 'ghost_box', 'ghost_box_push']) {
      if (bodyName !== activeObject && bodyName !== activeGhost) {
        this._setFreeJointBodyPose(bodyName, farPos);
      }
    }
  }

  _applyBoxHalfDimsToScene(halfDims, task = null) {
    if (!this.model || !halfDims) {
      return;
    }
    const normalizedTask = normalizePlannerTask(task ?? this._activePlannerTask());
    const dims = Array.from(halfDims, Number);
    if (this.policyRunner) {
      this.policyRunner.boxHalfDims = new Float32Array(dims);
      this.policyRunner.bboxOffsets = [
        [dims[0], dims[1], dims[2]], [dims[0], dims[1], -dims[2]],
        [dims[0], -dims[1], dims[2]], [dims[0], -dims[1], -dims[2]],
        [-dims[0], dims[1], dims[2]], [-dims[0], dims[1], -dims[2]],
        [-dims[0], -dims[1], dims[2]], [-dims[0], -dims[1], -dims[2]]
      ];
      this.policyRunner.objectBodyName = plannerObjectBodyName(normalizedTask);
    }
    this._moveInactivePlannerBoxesAway(normalizedTask);
    this.simulation?.forward?.();
  }

  _applyPlannerTaskGeometry(task) {
    this._applyBoxHalfDimsToScene(this._boxHalfDimsForTask(task), task);
  }

  _uprightBoxZAxisIfNeeded(context = 'auto_replan') {
    const activeBox = this._activePlannerObjectBodyName();
    const quat = this._bodyMjQuat(activeBox);
    const adr = this._freeJointAdrForBody(activeBox);
    if (!quat || !adr || adr.qpos < 0 || !this.simulation?.qpos) {
      return null;
    }
    const zDot = this._quatLocalZWorld(quat)[2];
    if (Math.abs(zDot) >= this.autoReplanConfig.uprightZThreshold) {
      if (this.policyRunner) {
        this.policyRunner.objectQuatOverride = null;
      }
      return null;
    }

    const uprightQuat = this._uprightYawQuatFrom(quat);
    this.simulation.qpos.set(uprightQuat, adr.qpos + 3);
    if (this.simulation.qvel && adr.qvel >= 0) {
      this.simulation.qvel.fill(0, adr.qvel, adr.qvel + 6);
    }
    this.simulation.forward();
    if (this.policyRunner) {
      this.policyRunner.objectQuatOverride = Float32Array.from(uprightQuat);
    }
    console.log(`[closed_loop] upright box before ${context}: z_dot=${zDot.toFixed(3)}, quat=${uprightQuat.map((v) => v.toFixed(4)).join(',')}`);
    return uprightQuat;
  }

  _maybeClosedLoopReplan() {
    if (!this.autoReplanEnabled || this.isInteractionMode || !this.policyRunner || this.policyRunner.referenceSource !== 'rule_planner') {
      return;
    }

    const status = this._isBoxHeldForReplan();
    const nearGoal = this._boxNearGoalForReplan();

    if (status.held) {
      this.autoReplanState.hasHeldBox = true;
      this.autoReplanState.detachCounter = 0;
      this.autoReplanState.refMismatchCounter = 0;
      this.autoReplanState.pendingReplan = false;
      this.autoReplanState.pendingReason = '';
      this.autoReplanState.staticCounter = 0;
      return;
    }

    if (this.autoReplanState.cooldownCounter > 0) {
      this.autoReplanState.cooldownCounter -= 1;
      return;
    }

    const speed = this._boxSpeedForReplan();
    const refError = this._boxReferenceErrorForReplan();
    const refMismatch = refError >= this.autoReplanConfig.refMismatchThreshold;

    if ((!this.autoReplanState.hasHeldBox && !refMismatch) || nearGoal) {
      this.autoReplanState.detachCounter = 0;
      this.autoReplanState.refMismatchCounter = 0;
      this.autoReplanState.pendingReplan = false;
      this.autoReplanState.pendingReason = '';
      this.autoReplanState.staticCounter = 0;
      return;
    }

    if (this.autoReplanState.pendingReplan) {
      const isStatic = speed.linear <= this.autoReplanConfig.objSpeedThreshold
        && speed.angular <= this.autoReplanConfig.objAngSpeedThreshold;
      this.autoReplanState.staticCounter = isStatic ? this.autoReplanState.staticCounter + 1 : 0;
      if (this.autoReplanState.staticCounter >= this.autoReplanConfig.staticTicks) {
        this._uprightBoxZAxisIfNeeded('drop_box_wait_static');
        this.replanCarryBoxFromCurrentState({
          reason: this.autoReplanState.pendingReason || 'drop_box_wait_static',
          minDist: status.minDist,
          inContact: status.inContact,
          objLinSpeed: speed.linear,
          objAngSpeed: speed.angular
        });
      }
      return;
    }

    if (this.autoReplanState.hasHeldBox) {
      this.autoReplanState.detachCounter += 1;
    } else {
      this.autoReplanState.detachCounter = 0;
    }
    if (refMismatch) {
      this.autoReplanState.refMismatchCounter += 1;
    } else {
      this.autoReplanState.refMismatchCounter = 0;
    }

    const dropTriggered = this.autoReplanState.detachCounter >= this.autoReplanConfig.detachTriggerTicks;
    const movedBeforeHoldTriggered = this.autoReplanState.refMismatchCounter >= this.autoReplanConfig.refMismatchTriggerTicks;
    if (dropTriggered || movedBeforeHoldTriggered) {
      this.autoReplanState.pendingReplan = true;
      this.autoReplanState.pendingReason = dropTriggered ? 'drop_box_wait_static' : 'box_ref_mismatch_wait_static';
      this.autoReplanState.staticCounter = 0;
      console.log(
        `[closed_loop] ${dropTriggered ? 'drop' : 'box-ref mismatch'} detected, waiting for object to settle | `
        + `ref_err=${refError.toFixed(3)}/${this.autoReplanConfig.refMismatchThreshold.toFixed(3)} m, `
        + `lin=${speed.linear.toFixed(3)}/${this.autoReplanConfig.objSpeedThreshold.toFixed(3)} m/s, `
        + `ang=${speed.angular.toFixed(3)}/${this.autoReplanConfig.objAngSpeedThreshold.toFixed(3)} rad/s`
      );
    }
  }

  updateFollowBodyId() {
    if (Number.isInteger(this.pelvis_body_id)) {
      this.followBodyId = this.pelvis_body_id;
      return;
    }
    if (this.model && this.model.nbody > 1) {
      this.followBodyId = 1;
    }
  }

  updateCameraFollow() {
    if (!this.followEnabled) {
      return;
    }
    const bodyId = Number.isInteger(this.followBodyId) ? this.followBodyId : null;
    if (bodyId === null) {
      return;
    }
    const cached = this.lastSimState.bodies.get(bodyId);
    if (!cached) {
      return;
    }
    this.followTargetDesired.set(cached.position.x, this.followHeight, cached.position.z);
    if (!this.followInitialized) {
      this.followTarget.copy(this.followTargetDesired);
      this.followInitialized = true;
    } else {
      this.followTarget.lerp(this.followTargetDesired, this.followLerp);
    }

    this.followDelta.subVectors(this.followTarget, this.controls.target);
    this.controls.target.copy(this.followTarget);
    this.camera.position.add(this.followDelta);
  }

  applyPolicyReferenceMocap() {
    if (!this.policyRunner?.currentReferenceFrame) {
      return;
    }

    const frame = this.policyRunner.currentReferenceFrame();
    if (!frame) {
      return;
    }

    const hasMocapBuffers = !!(this.bodyNameToMocapId && this.simulation?.mocap_pos && this.simulation?.mocap_quat);
    const mocapPos = hasMocapBuffers ? this.simulation.mocap_pos : null;
    const mocapQuat = hasMocapBuffers ? this.simulation.mocap_quat : null;
    const setMocap = (bodyName, pose, fallbackQuat = [1, 0, 0, 0]) => {
      if (!pose) {
        return;
      }
      const pos = Array.isArray(pose) ? pose[0] : pose.pos;
      const quat = (Array.isArray(pose) ? pose[1] : pose.quat) ?? fallbackQuat;
      if (!pos) {
        return;
      }
      const mocapId = this.bodyNameToMocapId?.[bodyName];
      if (hasMocapBuffers && mocapId !== undefined && mocapId >= 0) {
        mocapPos.set(pos, mocapId * 3);
        mocapQuat.set(quat, mocapId * 4);
        return;
      }

      const bodyId = this.bodyNameToId?.[bodyName];
      const body = bodyId !== undefined ? this.bodies?.[bodyId] : null;
      if (body) {
        body.position.set(pos[0], pos[2], -pos[1]);
        body.quaternion.set(-quat[1], -quat[3], quat[2], -quat[0]);
      }
    };

    setMocap('ref_l_wrist_frame', frame.leftWrist);
    setMocap('ref_r_wrist_frame', frame.rightWrist);
    setMocap('ref_torso_frame', frame.torso);
    setMocap('ref_l_ankle_frame', frame.leftAnkle);
    setMocap('ref_r_ankle_frame', frame.rightAnkle);
    setMocap('plane_1_holder', frame.plane1 ? { pos: frame.plane1 } : null);
    setMocap('plane_2_holder', frame.plane2 ? { pos: frame.plane2 } : null);
    if (hasMocapBuffers) {
      this.simulation.forward();
    }
  }

  async main_loop() {
    if (!this.policyRunner && !this.interactionRunner) {
      return;
    }

    this.startMemoryMonitor();

    while (this.alive) {
      const loopStart = performance.now();

      const hasRunner = this.isInteractionMode ? !!this.interactionRunner : !!this.policyRunner;

      if (!this.params.paused && this.model && this.data && this.simulation && hasRunner) {
        const state = this.isInteractionMode ? this.readInteractionState() : this.readPolicyState();

        try {
          if (this.isInteractionMode) {
            this.actionTarget = await this.interactionRunner.step(state);
          } else {
            this.actionTarget = await this.policyRunner.step(state);
            this.applyPolicyReferenceMocap();
          }
        } catch (e) {
          console.error('Inference error in main loop:', e);
          this.alive = false;
          break;
        }

        // Cache simulation views once per frame to avoid creating
        // new TypedArray views on every property access
        const simQpos = this.simulation.qpos;
        const simQvel = this.simulation.qvel;
        const simCtrl = this.simulation.ctrl;
        const ctrlRange = this.model?.actuator_ctrlrange;

        for (let substep = 0; substep < this.decimation; substep++) {
          if (this.control_type === 'joint_position') {
            for (let i = 0; i < this.numActions; i++) {
              const qpos_adr = this.qpos_adr_policy[i];
              const qvel_adr = this.qvel_adr_policy[i];
              const ctrl_adr = this.ctrl_adr_policy[i];

              const targetJpos = this.actionTarget ? this.actionTarget[i] : 0.0;
              const kp = this.kpPolicy ? this.kpPolicy[i] : 0.0;
              const kd = this.kdPolicy ? this.kdPolicy[i] : 0.0;
              const torque = kp * (targetJpos - simQpos[qpos_adr]) + kd * (0 - simQvel[qvel_adr]);
              let ctrlValue = torque;
              if (ctrlRange && ctrlRange.length >= (ctrl_adr + 1) * 2) {
                const min = ctrlRange[ctrl_adr * 2];
                const max = ctrlRange[(ctrl_adr * 2) + 1];
                if (Number.isFinite(min) && Number.isFinite(max) && min < max) {
                  ctrlValue = Math.min(Math.max(ctrlValue, min), max);
                }
              }
              simCtrl[ctrl_adr] = ctrlValue;
            }
          } else if (this.control_type === 'torque') {
            console.error('Torque control not implemented yet.');
          }

          const simQfrcApplied = this.simulation.qfrc_applied;
          for (let i = 0; i < simQfrcApplied.length; i++) {
            simQfrcApplied[i] = 0.0;
          }

          const dragged = this.dragStateManager.physicsObject;
          if (dragged && dragged.bodyID) {
            const simXposDrag = this.simulation.xpos;
            const simXquatDrag = this.simulation.xquat;
            for (let b = 0; b < this.model.nbody; b++) {
              if (this.bodies[b]) {
                getPosition(simXposDrag, b, this.bodies[b].position);
                getQuaternion(simXquatDrag, b, this.bodies[b].quaternion);
                this.bodies[b].updateWorldMatrix();
              }
            }
            const bodyID = dragged.bodyID;
            this.dragStateManager.update();
            const force = toMujocoPos(
              this.dragStateManager.currentWorld.clone()
                .sub(this.dragStateManager.worldHit)
                .multiplyScalar(60.0)
            );
            // clamp force magnitude
            const forceMagnitude = Math.sqrt(force.x * force.x + force.y * force.y + force.z * force.z);
            const maxForce = 30.0;
            if (forceMagnitude > maxForce) {
              const scale = maxForce / forceMagnitude;
              force.x *= scale;
              force.y *= scale;
              force.z *= scale;
            }
            const point = toMujocoPos(this.dragStateManager.worldHit.clone());
            this.simulation.applyForce(force.x, force.y, force.z, 0, 0, 0, point.x, point.y, point.z, bodyID);
          }

          this.simulation.step();
          this._constrainPush2BoxMotion();
        }

        if (!this.isInteractionMode) {
          this._maybeClosedLoopReplan();
        }

        // Cache WASM views once for body/light update reads
        const simXpos = this.simulation.xpos;
        const simXquat = this.simulation.xquat;
        for (let b = 0; b < this.model.nbody; b++) {
          if (!this.bodies[b]) {
            continue;
          }
          if (!this.lastSimState.bodies.has(b)) {
            this.lastSimState.bodies.set(b, {
              position: new THREE.Vector3(),
              quaternion: new THREE.Quaternion()
            });
          }
          const cached = this.lastSimState.bodies.get(b);
          getPosition(simXpos, b, cached.position);
          getQuaternion(simXquat, b, cached.quaternion);
        }

        const numLights = this.model.nlight;
        const simLightXpos = this.simulation.light_xpos;
        const simLightXdir = this.simulation.light_xdir;
        for (let l = 0; l < numLights; l++) {
          if (!this.lights[l]) {
            continue;
          }
          if (!this.lastSimState.lights.has(l)) {
            this.lastSimState.lights.set(l, {
              position: new THREE.Vector3(),
              direction: new THREE.Vector3()
            });
          }
          const cached = this.lastSimState.lights.get(l);
          getPosition(simLightXpos, l, cached.position);
          getPosition(simLightXdir, l, cached.direction);
        }

        if (!this.lastSimState.tendons.numWraps || typeof this.lastSimState.tendons.numWraps !== 'object') {
          this.lastSimState.tendons.numWraps = { count: 0, matrix: this.lastSimState.tendons.matrix };
        }
        this.lastSimState.tendons.numWraps.count = this.model.nwrap;

        this._stepFrameCount += 1;
        const now = performance.now();
        const elapsedStep = now - this._stepLastTime;
        if (elapsedStep >= 500) {
          this.simStepHz = (this._stepFrameCount * 1000) / elapsedStep;
          this._stepFrameCount = 0;
          this._stepLastTime = now;
        }
      } else {
        this.simStepHz = 0;
        this._stepFrameCount = 0;
        this._stepLastTime = performance.now();
      }

      const loopEnd = performance.now();
      const elapsed = (loopEnd - loopStart) / 1000;
      const target = this.timestep * this.decimation;
      const sleepTime = Math.max(0, target - elapsed);
      await new Promise((resolve) => setTimeout(resolve, sleepTime * 1000));
    }
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setPixelRatio(this.renderScale);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this._lastRenderTime = 0;
    this.render();
  }

  setRenderScale(scale) {
    const clamped = Math.max(0.5, Math.min(2.0, scale));
    this.renderScale = clamped;
    this.renderer.setPixelRatio(this.renderScale);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this._lastRenderTime = 0;
    this.render();
  }

  async dispose() {
    this.alive = false;
    this.renderer.setAnimationLoop(null);
    this.stopMemoryMonitor();

    // Dispose ONNX sessions on policy runners (frees WASM memory)
    if (this.interactionRunner?.dispose) {
      try { await this.interactionRunner.dispose(); } catch (e) { /* ignore */ }
      this.interactionRunner = null;
    }
    if (this.policyRunner?.dispose) {
      try { await this.policyRunner.dispose(); } catch (e) { /* ignore */ }
      this.policyRunner = null;
    }

    // Dispose all geometries, materials, textures in the scene
    this.scene.traverse(obj => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) {
          obj.material.forEach(m => { if (m.map) m.map.dispose(); m.dispose(); });
        } else {
          if (obj.material.map) obj.material.map.dispose();
          obj.material.dispose();
        }
      }
    });

    this.renderer.dispose();
    this.controls.dispose();

    if (this.simulation) {
      this.simulation.free();
      this.simulation = null;
    }

    if (this._resizeHandler) {
      window.removeEventListener('resize', this._resizeHandler);
    }
  }

  getSimStepHz() {
    return this.simStepHz;
  }

  /** Lazily allocate reusable state buffers once numActions/sdfBodyIds are known. */
  _ensureStateBuffers() {
    if (this._stateJointPos && this._stateJointPos.length === this.numActions) return;
    this._stateJointPos = new Float32Array(this.numActions);
    this._stateJointVel = new Float32Array(this.numActions);
    this._stateRootPos = new Float32Array(3);
    this._stateRootQuat = new Float32Array(4);
    this._stateRootAngVel = new Float32Array(3);
    this._stateResult = { jointPos: null, jointVel: null, rootPos: null, rootQuat: null, rootAngVel: null };
    const numSdf = this.sdfBodyIds ? this.sdfBodyIds.length : 0;
    this._stateSdfPositions = new Float32Array(numSdf * 3);
    this._stateSdfQuaternions = new Float32Array(numSdf * 4);
    this._stateTmpPos = new THREE.Vector3();
    this._stateObjQuat = new Float32Array(4);
    this._stateInteractionResult = { jointPos: null, jointVel: null, rootPos: null, rootQuat: null, rootAngVel: null, sdfBodyPositions: null, sdfBodyQuaternions: null };
  }

  readPolicyState() {
    this._ensureStateBuffers();
    const qpos = this.simulation.qpos;
    const qvel = this.simulation.qvel;
    for (let i = 0; i < this.numActions; i++) {
      this._stateJointPos[i] = qpos[this.qpos_adr_policy[i]];
      this._stateJointVel[i] = qvel[this.qvel_adr_policy[i]];
    }
    const rqa = this.rootQposAdr ?? 0;
    const rva = this.rootQvelAdr ?? 0;
    this._stateRootPos[0] = qpos[rqa]; this._stateRootPos[1] = qpos[rqa + 1]; this._stateRootPos[2] = qpos[rqa + 2];
    this._stateRootQuat[0] = qpos[rqa + 3]; this._stateRootQuat[1] = qpos[rqa + 4]; this._stateRootQuat[2] = qpos[rqa + 5]; this._stateRootQuat[3] = qpos[rqa + 6];
    this._stateRootAngVel[0] = qvel[rva + 3]; this._stateRootAngVel[1] = qvel[rva + 4]; this._stateRootAngVel[2] = qvel[rva + 5];

    this._stateResult.jointPos = this._stateJointPos;
    this._stateResult.jointVel = this._stateJointVel;
    this._stateResult.rootPos = this._stateRootPos;
    this._stateResult.rootQuat = this._stateRootQuat;
    this._stateResult.rootAngVel = this._stateRootAngVel;
    return this._stateResult;
  }

  /**
   * Read the full interaction policy state including body positions
   * for SDF queries and scene object poses.
   */
  readInteractionState() {
    const base = this.readPolicyState();

    const numSdf = this.sdfBodyIds.length;
    const sdfBodyPositions = this._stateSdfPositions;
    const sdfBodyQuaternions = this._stateSdfQuaternions;
    const tmpPos = this._stateTmpPos;

    // Cache WASM views once instead of accessing per-body in loops
    const xpos = this.simulation.xpos;
    const xquat = this.simulation.xquat;

    for (let i = 0; i < numSdf; i++) {
      const bodyId = this.sdfBodyIds[i];
      getPosition(xpos, bodyId, tmpPos, false);
      sdfBodyPositions[i * 3] = tmpPos.x;
      sdfBodyPositions[i * 3 + 1] = tmpPos.y;
      sdfBodyPositions[i * 3 + 2] = tmpPos.z;
      getMujocoQuaternion(xquat, bodyId, sdfBodyQuaternions, i * 4);
    }

    // Update scene object poses in the SDF manager
    if (this.interactionRunner) {
      const objQuat = this._stateObjQuat;
      for (const [name, bodyId] of Object.entries(this.sceneObjectBodyIds)) {
        getPosition(xpos, bodyId, tmpPos, false);
        getMujocoQuaternion(xquat, bodyId, objQuat, 0);
        // setObjectPose expects array-like [0],[1],[2] indexing, not Vector3 .x,.y,.z
        this.interactionRunner.sdfManager.setObjectPose(name, [tmpPos.x, tmpPos.y, tmpPos.z], objQuat);
      }
    }

    this._stateInteractionResult.jointPos = base.jointPos;
    this._stateInteractionResult.jointVel = base.jointVel;
    this._stateInteractionResult.rootPos = base.rootPos;
    this._stateInteractionResult.rootQuat = base.rootQuat;
    this._stateInteractionResult.rootAngVel = base.rootAngVel;
    this._stateInteractionResult.sdfBodyPositions = sdfBodyPositions;
    this._stateInteractionResult.sdfBodyQuaternions = sdfBodyQuaternions;
    return this._stateInteractionResult;
  }

  /* ================== User-uploaded objects ================== */
  /** Tracks user-added objects: name -> { mesh, parsed, sdfData, scale } */
  _userObjects = {};
  _userObjectCounter = 0;

  /**
   * Compute SDF grid from parsed OBJ data at the given scale.
   * Scales a copy of the vertices so the grid covers the actual object size.
   */
  _computeScaledSdf(parsed, scale, gridSize = 32, onProgress) {
    let verts = parsed.vertices;
    if (scale !== 1.0) {
      verts = new Float64Array(parsed.vertices.length);
      for (let i = 0; i < verts.length; i++) {
        verts[i] = parsed.vertices[i] * scale;
      }
    }
    return computeSdfGrid(verts, parsed.faces, parsed.numFaces, gridSize, 1.0, onProgress);
  }

  /**
   * Add a user-uploaded OBJ to the scene.
   * Parses the OBJ text then delegates to _addParsedObject.
   * @param {string} objText - Raw OBJ file content
   * @param {object} opts - { name?, position?, color? }
   * @param {function} [onProgress] - SDF computation progress callback(face, total)
   * @returns {{ name: string }} identifier for the new object
   */
  addUserObject(objText, opts = {}, onProgress) {
    const parsed = parseObj(objText);
    if (parsed.numFaces === 0) {
      throw new Error('OBJ file contains no faces');
    }
    // Transform from OBJ space (Z-up, Y-forward) to MuJoCo space (Z-up, X-forward)
    transformObjToMujoco(parsed.vertices);
    return this._addParsedObject(parsed, opts, onProgress);
  }

  /**
   * Add a user-uploaded STL to the scene.
   * Parses the STL buffer then delegates to _addParsedObject.
   * @param {ArrayBuffer} buffer - Raw STL file contents
   * @param {object} opts - { name?, position?, color? }
   * @param {function} [onProgress] - SDF computation progress callback(face, total)
   * @returns {{ name: string }} identifier for the new object
   */
  addUserObjectFromStl(buffer, opts = {}, onProgress) {
    const parsed = parseStl(buffer);
    if (parsed.numFaces === 0) {
      throw new Error('STL file contains no faces');
    }
    // STL files are already in their native coordinate space.
    // Apply the same OBJ→MuJoCo transform (works on any Z-up mesh).
    transformObjToMujoco(parsed.vertices);
    return this._addParsedObject(parsed, opts, onProgress);
  }

  /**
   * Shared post-parse pipeline: center vertices, compute SDF, create Three.js mesh, register.
   * @param {object} parsed - { vertices, faces, numVerts, numFaces }
   * @param {object} opts - { name?, position?, color?, scale? }
   * @param {function} [onProgress]
   * @returns {{ name: string }}
   */
  _addParsedObject(parsed, opts = {}, onProgress) {
    // Center vertices at their centroid so the mesh origin = geometric center.
    const nVerts = parsed.numVerts;
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < parsed.vertices.length; i += 3) {
      cx += parsed.vertices[i];
      cy += parsed.vertices[i + 1];
      cz += parsed.vertices[i + 2];
    }
    cx /= nVerts; cy /= nVerts; cz /= nVerts;
    for (let i = 0; i < parsed.vertices.length; i += 3) {
      parsed.vertices[i] -= cx;
      parsed.vertices[i + 1] -= cy;
      parsed.vertices[i + 2] -= cz;
    }

    const name = opts.name || `user_obj_${++this._userObjectCounter}`;
    const pos = opts.position || [0.7, 0.7, 0.2];
    const scale = opts.scale ?? 1.0;
    const color = opts.color ?? 0x3b82f6;

    // SDF computation is deferred to confirmUserObject() to avoid expensive
    // computation during preview.  Only the Three.js visual is created here.

    // Create THREE.js visual mesh
    const geometry = buildThreeGeometry(parsed.vertices, parsed.faces, parsed.numFaces, THREE);
    const material = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(color),
      roughness: 0.4,
      metalness: 0.1,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    // Place in MuJoCo coordinate frame -> THREE.js (swap Y/Z)
    mesh.position.set(pos[0], pos[2], -pos[1]);
    mesh.scale.set(scale, scale, scale);

    const mujocoRoot = this.scene.getObjectByName('MuJoCo Root');
    if (mujocoRoot) {
      mujocoRoot.add(mesh);
    } else {
      this.scene.add(mesh);
    }

    this._userObjects[name] = { mesh, parsed, sdfData: null, scale };
    return { name };
  }

  /**
   * Update position of a user-uploaded object.
   * @param {string} name
   * @param {number[]} pos - [x, y, z] in MuJoCo world frame
   */
  setUserObjectPose(name, pos) {
    const obj = this._userObjects[name];
    if (!obj) return;

    // Update THREE.js visual (swap Y/Z for MuJoCo -> THREE.js)
    obj.mesh.position.set(pos[0], pos[2], -pos[1]);

    // Update SDF manager pose
    if (this.interactionRunner) {
      this.interactionRunner.sdfManager.setObjectPose(name, pos, [1, 0, 0, 0]);
    }
  }

  /**
   * Update uniform scale of a user-uploaded object.
   * @param {string} name
   * @param {number} scale
   */
  setUserObjectScale(name, scale) {
    const obj = this._userObjects[name];
    if (!obj) return;

    // Update THREE.js visual only; SDF is computed at confirm time
    obj.mesh.scale.set(scale, scale, scale);
    obj.scale = scale;
  }

  /**
   * Update rotation of a user-uploaded object via Euler angles.
   * @param {string} name
   * @param {number[]} euler - [rx, ry, rz] in radians (XYZ extrinsic)
   */
  setUserObjectRotation(name, euler) {
    const obj = this._userObjects[name];
    if (!obj) return;

    const quat = eulerToQuat(euler[0], euler[1], euler[2]);

    // Apply the same swizzle as getQuaternion in mujocoUtils.js:
    //   target.set(-buffer[1], -buffer[3], buffer[2], -buffer[0])
    // Maps MuJoCo (w,x,y,z) to THREE.js (x,y,z,w) = (-mj_x, -mj_z, mj_y, -mj_w)
    obj.mesh.quaternion.set(-quat[1], -quat[3], quat[2], -quat[0]);

    // Update SDF manager pose (keep position, update quat in MuJoCo frame)
    if (this.interactionRunner) {
      const pose = this.interactionRunner.sdfManager.objectPoses.get(name);
      if (pose) {
        pose.quat[0] = quat[0];
        pose.quat[1] = quat[1];
        pose.quat[2] = quat[2];
        pose.quat[3] = quat[3];
      }
    }
  }

  /**
   * Update both position and rotation of a user-uploaded object.
   * @param {string} name
   * @param {number[]} pos - [x, y, z] in MuJoCo world frame
   * @param {number[]} euler - [rx, ry, rz] in radians (XYZ extrinsic)
   */
  setUserObjectPoseAndRotation(name, pos, euler) {
    const obj = this._userObjects[name];
    if (!obj) return;

    const quat = eulerToQuat(euler[0], euler[1], euler[2]);

    // Update THREE.js position (swap Y/Z for MuJoCo -> THREE.js)
    obj.mesh.position.set(pos[0], pos[2], -pos[1]);
    // Update THREE.js rotation (same swizzle as getQuaternion)
    obj.mesh.quaternion.set(-quat[1], -quat[3], quat[2], -quat[0]);

    // Update SDF manager
    if (this.interactionRunner) {
      this.interactionRunner.sdfManager.setObjectPose(name, pos, quat);
    }
  }

  /**
   * Confirm a user-uploaded object: inject it as a MuJoCo rigid body with
   * a mesh collision geom using the actual OBJ mesh (convex hull).
   * Reloads the scene and re-initializes the policy.
   * @param {string} name
   * @param {number[]} pos - [x, y, z] in MuJoCo world frame
   * @param {number[]} euler - [rx, ry, rz] in radians
   * @param {number} scale
   * @returns {Promise<boolean>} true if successful
   */
  async confirmUserObject(name, pos, euler, scale, mass = 1.0, friction = 1.0, sdfResolution = 32, onProgress) {
    const obj = this._userObjects[name];
    if (!obj) return false;

    // Compute SDF at the final scale (deferred from upload time)
    console.log(`Computing SDF for "${name}" (${obj.parsed.numFaces} triangles, scale=${scale}, grid=${sdfResolution})...`);
    // Yield to UI before heavy computation
    await new Promise(r => setTimeout(r, 50));
    const sdfData = this._computeScaledSdf(obj.parsed, scale, sdfResolution, onProgress);
    obj.sdfData = sdfData;
    obj.scale = scale;
    console.log(`SDF computed for "${name}"`);

    const quat = eulerToQuat(euler[0], euler[1], euler[2]);

    // Write the OBJ mesh as binary STL to MuJoCo filesystem
    const stlData = objToStlBinary(obj.parsed.vertices, obj.parsed.faces, obj.parsed.numFaces);
    // Ensure meshes directory exists
    if (!this.mujoco.FS.analyzePath('/working/g1/meshes').exists) {
      this.mujoco.FS.mkdir('/working/g1/meshes');
    }
    this.mujoco.FS.writeFile(`/working/g1/meshes/${name}.stl`, stlData);
    console.log(`Wrote STL for "${name}": ${stlData.length} bytes`);

    // Compute bounding box from parsed OBJ vertices for inertia estimation
    const verts = obj.parsed.vertices;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < verts.length; i += 3) {
      minX = Math.min(minX, verts[i]); maxX = Math.max(maxX, verts[i]);
      minY = Math.min(minY, verts[i+1]); maxY = Math.max(maxY, verts[i+1]);
      minZ = Math.min(minZ, verts[i+2]); maxZ = Math.max(maxZ, verts[i+2]);
    }
    // Half-extents in MuJoCo coords (after scale); vertices already in MuJoCo space
    const hx = (maxX - minX) * scale * 0.5;
    const hy = (maxY - minY) * scale * 0.5;
    const hz = (maxZ - minZ) * scale * 0.5;
    // Solid box inertia approximation: I = m/12 * (b^2 + c^2)
    const ixx = mass / 12 * (4*hy*hy + 4*hz*hz);
    const iyy = mass / 12 * (4*hx*hx + 4*hz*hz);
    const izz = mass / 12 * (4*hx*hx + 4*hy*hy);
    const minI = 0.001;

    // Build MJCF body XML with explicit inertial using user-specified mass/friction
    const bodyXml = `
        <body name="${name}" pos="${pos[0].toFixed(4)} ${pos[1].toFixed(4)} ${pos[2].toFixed(4)}" quat="${quat[0].toFixed(6)} ${quat[1].toFixed(6)} ${quat[2].toFixed(6)} ${quat[3].toFixed(6)}">
            <freejoint name="${name}_joint"/>
            <inertial pos="0 0 0" mass="${mass.toFixed(3)}" diaginertia="${Math.max(ixx,minI).toFixed(6)} ${Math.max(iyy,minI).toFixed(6)} ${Math.max(izz,minI).toFixed(6)}"/>
            <geom name="${name}_geom" type="mesh" mesh="${name}_mesh"
                  rgba="0.23 0.51 0.96 1"
                  contype="1" conaffinity="1" friction="${friction.toFixed(2)} 0.005 0.0001"/>
        </body>`;

    // Read the current scene XML from MuJoCo FS
    const sceneFile = this._currentSceneFile || 'g1/g1_interaction.xml';
    let xml;
    try {
      xml = this.mujoco.FS.readFile('/working/' + sceneFile, { encoding: 'utf8' });
    } catch (e) {
      console.error('Failed to read scene XML:', e);
      return false;
    }

    // 1. Inject mesh asset into <asset> section
    const assetEndIdx = xml.indexOf('</asset>');
    if (assetEndIdx < 0) {
      console.error('Could not find </asset> in scene XML');
      return false;
    }
    const meshAssetXml = `\n        <mesh name="${name}_mesh" file="${name}.stl" scale="${scale} ${scale} ${scale}" refpos="0 0 0" refquat="1 0 0 0"/>`;
    xml = xml.slice(0, assetEndIdx) + meshAssetXml + '\n    ' + xml.slice(assetEndIdx);

    // 2. Inject body before the robot body (find insertion point AFTER asset edit)
    const insertMarker = '<!-- ================= G1 Robot ================= -->';
    let insertIdx = xml.indexOf(insertMarker);
    if (insertIdx < 0) {
      insertIdx = xml.indexOf('</worldbody>');
    }
    if (insertIdx < 0) {
      console.error('Could not find injection point in scene XML');
      return false;
    }
    xml = xml.slice(0, insertIdx) + bodyXml + '\n        ' + xml.slice(insertIdx);

    // Save all confirmed user objects' SDF data to re-register after reload
    const userObjSdfBackup = {};
    for (const [n, o] of Object.entries(this._userObjects)) {
      userObjSdfBackup[n] = {
        sdfData: o.sdfData,
        parsed: o.parsed,
        confirmed: o.confirmed || false,
        scale: o.scale || 1.0,
      };
    }
    // Store scale on the current object being confirmed
    obj.scale = scale;
    userObjSdfBackup[name].scale = scale;
    // Mark this object as confirmed
    obj.confirmed = true;
    userObjSdfBackup[name].confirmed = true;

    // Remove visual-only meshes (MuJoCo will render the mesh geom after reload)
    for (const o of Object.values(this._userObjects)) {
      if (o.mesh && o.mesh.parent) {
        o.mesh.parent.remove(o.mesh);
      }
    }

    // Pause and reload
    this.alive = false;
    this.params.paused = true;

    // Write modified XML
    const modifiedFile = 'g1/g1_interaction_modified.xml';
    this.mujoco.FS.writeFile('/working/' + modifiedFile, xml);
    this._currentSceneFile = modifiedFile;

    try {
      // Reload scene
      await this.reloadScene(modifiedFile);
      this.updateFollowBodyId();

      // Reload interaction policy
      const configResp = await fetch(defaultPolicy);
      const configJson = await configResp.json();
      await this.reloadInteractionPolicy(defaultPolicy, configJson);

      // Re-register all user objects' SDF grids (scale=1.0 since SDF precomputed at scaled size)
      for (const [n, backup] of Object.entries(userObjSdfBackup)) {
        const grid = new SDFVoxelGrid(backup.sdfData);
        this.interactionRunner.sdfManager.grids.set(n, grid);
        this.interactionRunner.sdfManager.objectPoses.set(n, {
          pos: new Float32Array([0, 0, 0]),
          quat: new Float32Array([1, 0, 0, 0]),
          scale: 1.0,
        });
      }

      // Register confirmed objects in sceneObjectBodyIds so their
      // MuJoCo poses are tracked and fed back to the SDF manager
      for (const [n, backup] of Object.entries(userObjSdfBackup)) {
        if (backup.confirmed && this.bodyNameToId[n] !== undefined) {
          this.sceneObjectBodyIds[n] = this.bodyNameToId[n];
        }
      }

      // Replace auto-generated material on confirmed objects with custom material
      for (const [n, backup] of Object.entries(userObjSdfBackup)) {
        if (!backup.confirmed) continue;
        const bodyId = this.bodyNameToId[n];
        if (bodyId === undefined || !this.bodies[bodyId]) continue;

        this.bodies[bodyId].traverse(child => {
          if (child.isMesh && child.material) {
            child.material.dispose();
            child.material = new THREE.MeshPhysicalMaterial({
              color: new THREE.Color(0x3b82f6),
              roughness: 0.4,
              metalness: 0.1,
            });
          }
        });

        this._userObjects[n] = {
          parsed: backup.parsed,
          sdfData: backup.sdfData,
          confirmed: true,
          scale: backup.scale || 1.0,
        };
      }

      // Restart simulation loop
      this.alive = true;
      this.params.paused = false;
      this.main_loop();

      return true;
    } catch (e) {
      console.error('Failed to confirm user object:', e);
      this.alive = true;
      this.params.paused = false;
      this.main_loop();
      return false;
    }
  }

  /**
   * Remove a user-uploaded object.
   * For unconfirmed objects, removes the visual mesh and SDF data.
   * For confirmed objects, removes the body from the MuJoCo scene XML and reloads.
   * @param {string} name
   * @returns {Promise<boolean>} true if successful
   */
  async removeUserObject(name) {
    const obj = this._userObjects[name];
    if (!obj) return false;

    // Remove from SDF manager
    if (this.interactionRunner) {
      this.interactionRunner.sdfManager.grids.delete(name);
      this.interactionRunner.sdfManager.objectPoses.delete(name);
    }

    // For unconfirmed objects: just remove the visual mesh
    if (!obj.confirmed) {
      if (obj.mesh) {
        if (obj.mesh.parent) {
          obj.mesh.parent.remove(obj.mesh);
        }
        obj.mesh.geometry.dispose();
        obj.mesh.material.dispose();
      }
      delete this._userObjects[name];
      return true;
    }

    // --- Confirmed object: remove from scene XML and reload ---
    delete this._userObjects[name];
    delete this.sceneObjectBodyIds[name];

    const sceneFile = this._currentSceneFile || 'g1/g1_interaction.xml';
    let xml;
    try {
      xml = this.mujoco.FS.readFile('/working/' + sceneFile, { encoding: 'utf8' });
    } catch (e) {
      console.error('Failed to read scene XML for removal:', e);
      return false;
    }

    // Remove body XML block: <body name="NAME" ...>...</body>
    const bodyRegex = new RegExp(`\\s*<body\\s+name="${name}"[^>]*>[\\s\\S]*?</body>`, 'g');
    xml = xml.replace(bodyRegex, '');

    // Remove mesh asset: <mesh name="NAME_mesh" .../>
    const meshRegex = new RegExp(`\\s*<mesh\\s+name="${name}_mesh"[^/]*/>`);
    xml = xml.replace(meshRegex, '');

    // Remove STL file from MuJoCo filesystem
    try {
      this.mujoco.FS.unlink(`/working/g1/meshes/${name}.stl`);
    } catch (e) { /* file may not exist */ }

    // Backup remaining confirmed objects' SDF data
    const remainingBackup = {};
    for (const [n, o] of Object.entries(this._userObjects)) {
      if (o.confirmed) {
        remainingBackup[n] = {
          sdfData: o.sdfData,
          parsed: o.parsed,
          scale: o.scale || 1.0,
        };
      }
    }

    // Pause and reload
    this.alive = false;
    this.params.paused = true;

    const modifiedFile = 'g1/g1_interaction_modified.xml';
    this.mujoco.FS.writeFile('/working/' + modifiedFile, xml);
    this._currentSceneFile = modifiedFile;

    try {
      await this.reloadScene(modifiedFile);
      this.updateFollowBodyId();

      const configResp = await fetch(defaultPolicy);
      const configJson = await configResp.json();
      await this.reloadInteractionPolicy(defaultPolicy, configJson);

      // Re-register remaining confirmed objects' SDF grids (scale=1.0, already at scaled size)
      for (const [n, backup] of Object.entries(remainingBackup)) {
        const grid = new SDFVoxelGrid(backup.sdfData);
        this.interactionRunner.sdfManager.grids.set(n, grid);
        this.interactionRunner.sdfManager.objectPoses.set(n, {
          pos: new Float32Array([0, 0, 0]),
          quat: new Float32Array([1, 0, 0, 0]),
          scale: 1.0,
        });
      }

      // Re-register body IDs for remaining confirmed objects
      for (const n of Object.keys(remainingBackup)) {
        if (this.bodyNameToId[n] !== undefined) {
          this.sceneObjectBodyIds[n] = this.bodyNameToId[n];
        }
      }

      // Re-apply custom material on remaining confirmed objects
      for (const n of Object.keys(remainingBackup)) {
        const bodyId = this.bodyNameToId[n];
        if (bodyId === undefined || !this.bodies[bodyId]) continue;
        this.bodies[bodyId].traverse(child => {
          if (child.isMesh && child.material) {
            child.material.dispose();
            child.material = new THREE.MeshPhysicalMaterial({
              color: new THREE.Color(0x3b82f6),
              roughness: 0.4,
              metalness: 0.1,
            });
          }
        });
      }

      this.alive = true;
      this.params.paused = false;
      this.main_loop();
      return true;
    } catch (e) {
      console.error('Failed to remove confirmed object:', e);
      this.alive = true;
      this.params.paused = false;
      this.main_loop();
      return false;
    }
  }

  /** @returns {string[]} names of all user objects */
  getUserObjectNames() {
    return Object.keys(this._userObjects);
  }

  resetSimulation() {
    if (!this.simulation) {
      return;
    }
    this.params.paused = true;
    this.simulation.resetData();
    this.simulation.forward();
    this.actionTarget = null;
    if (this.isInteractionMode && this.interactionRunner) {
      const state = this.readPolicyState();
      this.interactionRunner.reset(state);
    } else if (this.policyRunner) {
      const state = this.readPolicyState();
      this.policyRunner.reset(state);
      this.params.current_motion = 'default';
    }
    if (this.policyRunner) {
      this.policyRunner.objectQuatOverride = null;
    }
    this.resetAutoReplanState();
    this.params.paused = false;
  }

  async applyCarryBoxPlannerConfig(config = {}) {
    const rawBoxStartPos = config.boxStartPos ?? config.box_start_pos ?? null;
    const rawGoalPos = config.goalPos ?? config.goal_pos ?? null;
    const task = normalizePlannerTask(config.task ?? config.plannerTask ?? config.planner_task ?? 'carrybox');
    const taskHalfDims = this._boxHalfDimsForTask(task);
    const { boxStartPos, goalPos } = enforcePlannerObjectHeights(task, rawBoxStartPos, rawGoalPos, taskHalfDims);
    const shouldResetSimulation = Boolean(config.resetSimulation ?? config.reset_simulation);
    if (!this.simulation || !this.model) {
      this.carryBoxPlannerConfig = {
        boxStartPos: boxStartPos ? Array.from(boxStartPos) : null,
        goalPos: goalPos ? Array.from(goalPos) : null,
        task
      };
      return;
    }

    this.params.paused = true;
    await this._ensurePlannerSceneForTask(task);
    if (shouldResetSimulation) {
      this.simulation.resetData();
    }
    const stored = this.carryBoxPlannerConfig ?? {};
    this.carryBoxPlannerConfig = {
      boxStartPos: boxStartPos ? Array.from(boxStartPos) : stored.boxStartPos ?? null,
      goalPos: goalPos ? Array.from(goalPos) : stored.goalPos ?? null,
      task
    };
    this._applyPlannerTaskGeometry(this.carryBoxPlannerConfig.task);

    if (this.policyRunner && this.carryBoxPlannerConfig.goalPos) {
      this.policyRunner.goalPos = new Float32Array(this.carryBoxPlannerConfig.goalPos);
      this.policyRunner.task = this.carryBoxPlannerConfig.task;
      this.policyRunner.objectQuatOverride = null;
    }

    if (this.carryBoxPlannerConfig.boxStartPos) {
      this._setFreeJointBodyPose(
        this._activePlannerObjectBodyName(this.carryBoxPlannerConfig.task),
        this.carryBoxPlannerConfig.boxStartPos
      );
      this._moveInactivePlannerBoxesAway(this.carryBoxPlannerConfig.task);
    }

    this.simulation.forward();
    if (this.policyRunner) {
      const state = this.readPolicyState();
      this.policyRunner.reset(state);
      this.applyPolicyReferenceMocap();
    }
    this.actionTarget = null;
    this.resetAutoReplanState();
    this.params.paused = false;
  }

  updateCarryBoxReplanGoal(config = {}) {
    const rawGoalPos = config.goalPos ?? config.goal_pos ?? null;
    const taskArg = config.task ?? config.plannerTask ?? config.planner_task ?? null;
    if (!rawGoalPos && !taskArg) {
      return;
    }
    const stored = this.carryBoxPlannerConfig ?? {};
    const task = taskArg ? normalizePlannerTask(taskArg) : stored.task ?? this.policyRunner?.task ?? 'carrybox';
    const baseGoalPos = rawGoalPos ?? stored.goalPos ?? null;
    const { goalPos } = enforcePlannerObjectHeights(task, null, baseGoalPos, this._boxHalfDimsForTask(task));
    this.carryBoxPlannerConfig = {
      ...stored,
      goalPos: goalPos ? Array.from(goalPos) : stored.goalPos ?? null,
      task
    };
    this._applyPlannerTaskGeometry(this.carryBoxPlannerConfig.task);
    if (this.policyRunner) {
      if (this.carryBoxPlannerConfig.goalPos) {
        this.policyRunner.goalPos = new Float32Array(this.carryBoxPlannerConfig.goalPos);
      }
      this.policyRunner.task = this.carryBoxPlannerConfig.task;
    }
  }

  replanCarryBoxFromCurrentState(config = {}) {
    const rawGoalPos = config.goalPos ?? config.goal_pos ?? null;
    const taskArg = config.task ?? config.plannerTask ?? config.planner_task ?? null;
    if (!this.simulation || !this.model || !this.policyRunner) {
      return;
    }

    this.params.paused = true;
    const stored = this.carryBoxPlannerConfig ?? {};
    const task = taskArg ? normalizePlannerTask(taskArg) : stored.task ?? this.policyRunner.task ?? 'carrybox';
    const baseGoalPos = rawGoalPos ?? stored.goalPos ?? null;
    const { goalPos } = enforcePlannerObjectHeights(task, null, baseGoalPos, this._boxHalfDimsForTask(task));
    this._applyPlannerTaskGeometry(task);
    if (goalPos) {
      this.carryBoxPlannerConfig = {
        ...stored,
        goalPos: Array.from(goalPos),
        task
      };
      this.policyRunner.goalPos = new Float32Array(this.carryBoxPlannerConfig.goalPos);
      this.policyRunner.task = this.carryBoxPlannerConfig.task;
    } else if (taskArg) {
      this.carryBoxPlannerConfig = {
        ...stored,
        task
      };
      this.policyRunner.task = this.carryBoxPlannerConfig.task;
    }

    this.simulation.forward();
    const state = this.readPolicyState();
    this.policyRunner.reset(state);
    this.applyPolicyReferenceMocap();
    this.actionTarget = null;
    this.params.current_motion = 'default';
    this.autoReplanState.hasHeldBox = false;
    this.autoReplanState.detachCounter = 0;
    this.autoReplanState.refMismatchCounter = 0;
    this.autoReplanState.cooldownCounter = this.autoReplanConfig.replanCooldownTicks;
    this.autoReplanState.pendingReplan = false;
    this.autoReplanState.pendingReason = '';
    this.autoReplanState.staticCounter = 0;
    this.autoReplanState.replanCounter += 1;
    this.autoReplanState.lastReason = config.reason ?? 'manual_replan';
    console.log(
      `[closed_loop] replan#${this.autoReplanState.replanCounter} | reason=${this.autoReplanState.lastReason}, `
      + `min_palm_dist=${Number(config.minDist ?? config.min_dist ?? NaN).toFixed(3)}, `
      + `hand_contact=${Number(Boolean(config.inContact ?? config.in_contact))}, `
      + `obj_lin_speed=${Number(config.objLinSpeed ?? config.obj_lin_speed ?? NaN).toFixed(3)} m/s, `
      + `obj_ang_speed=${Number(config.objAngSpeed ?? config.obj_ang_speed ?? NaN).toFixed(3)} rad/s`
    );
    this.params.paused = false;
  }

  render() {
    if (!this.model || !this.data || !this.simulation) {
      return;
    }
    const now = performance.now();
    if (now - this._lastRenderTime < 30) {
      return;
    }
    this._lastRenderTime = now;

    this.updateCameraFollow();
    this.controls.update();

    for (const [b, cached] of this.lastSimState.bodies) {
      if (this.bodies[b]) {
        this.bodies[b].position.copy(cached.position);
        this.bodies[b].quaternion.copy(cached.quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }

    // Update root command visualization (using pre-allocated temps)
    if (this.isInteractionMode && this.interactionRunner && this.pelvis_body_id != null) {
      const pelvisCached = this.lastSimState.bodies.get(this.pelvis_body_id);
      if (pelvisCached) {
        const tp = this.interactionRunner.targetRootPos;
        const targetThree = this._tmpTargetThree.set(tp[0], tp[2], -tp[1]);
        const robotPos = this._tmpRobotPos.copy(pelvisCached.position);
        const groundY = 0.02;
        robotPos.y = groundY;
        targetThree.y = groundY;

        const dir = this._tmpDir.copy(targetThree).sub(robotPos);
        const len = dir.length();

        if (len > 0.05) {
          this.rootCmdArrow.visible = true;
          this.rootCmdArrow.position.copy(robotPos);
          this.rootCmdArrow.setDirection(dir.normalize());
          this.rootCmdArrow.setLength(len, Math.min(len * 0.25, 0.15), 0.06);
        } else {
          this.rootCmdArrow.visible = false;
        }

        // Heading indicator at target
        const tq = this.interactionRunner.targetRootQuat;
        const w = tq[0], qx = tq[1], qy = tq[2], qz = tq[3];
        const headingDir = this._tmpHeadingDir.set(
          1 - 2 * (qy * qy + qz * qz),
          2 * (qx * qz - w * qy),
          -(2 * (qx * qy + w * qz))
        ).normalize();
        headingDir.y = 0; headingDir.normalize();

        this.rootCmdHeading.visible = true;
        this.rootCmdHeading.position.copy(targetThree);
        this.rootCmdHeading.setDirection(headingDir);
        this.rootCmdHeading.setLength(0.4, 0.1, 0.05);
      }
    } else {
      this.rootCmdArrow.visible = false;
      this.rootCmdHeading.visible = false;
    }

    // Update SDF visualization arrows
    if (this.showSdfVis && this.isInteractionMode && this.interactionRunner) {
      const runner = this.interactionRunner;
      const numSdf = runner.numSdfBodies;
      this._ensureSdfArrows(numSdf);

      for (let i = 0; i < numSdf; i++) {
        const dist = runner.sdfVisDistances[i];
        // Gradient in MuJoCo world frame (points away from surface)
        const gx = runner.sdfVisGradients[i * 3];
        const gy = runner.sdfVisGradients[i * 3 + 1];
        const gz = runner.sdfVisGradients[i * 3 + 2];
        // Position in MuJoCo world frame
        const px = runner.sdfVisPositions[i * 3];
        const py = runner.sdfVisPositions[i * 3 + 1];
        const pz = runner.sdfVisPositions[i * 3 + 2];

        // MuJoCo (x,y,z) → Three.js (x, z, -y)
        this._tmpSdfOrigin.set(px, pz, -py);

        // Arrow direction = opposite of gradient (points toward surface)
        // Gradient is in MuJoCo frame, apply same swizzle
        this._tmpSdfDir.set(-gx, -gz, gy);
        const dirLen = this._tmpSdfDir.length();

        const arrow = this._sdfArrows[i];
        if (dirLen > 1e-6 && dist > 1e-4 && dist < runner.maxValidDistance) {
          this._tmpSdfDir.divideScalar(dirLen);
          arrow.visible = true;
          arrow.position.copy(this._tmpSdfOrigin);
          arrow.setDirection(this._tmpSdfDir);
          const arrowLen = Math.min(dist, 1.0);
          arrow.setLength(arrowLen, Math.min(arrowLen * 0.2, 0.04), 0.02);
        } else {
          arrow.visible = false;
        }
      }
      // Hide extra arrows if count decreased
      for (let i = numSdf; i < this._sdfArrows.length; i++) {
        this._sdfArrows[i].visible = false;
      }
    } else {
      for (const arrow of this._sdfArrows) {
        arrow.visible = false;
      }
    }

    for (const [l, cached] of this.lastSimState.lights) {
      if (this.lights[l]) {
        this.lights[l].position.copy(cached.position);
        this.lights[l].lookAt(this._tmpLightTarget.copy(cached.direction).add(this.lights[l].position));
      }
    }

    if (this.mujocoRoot && this.mujocoRoot.cylinders) {
      const numWraps = this.lastSimState.tendons.numWraps.count;
      this.mujocoRoot.cylinders.count = numWraps;
      this.mujocoRoot.spheres.count = 0;
      this.mujocoRoot.spheres.visible = false;
      this.mujocoRoot.cylinders.instanceMatrix.needsUpdate = true;
      this.mujocoRoot.spheres.instanceMatrix.needsUpdate = true;
    }

    this.renderer.render(this.scene, this.camera);
  }
}
