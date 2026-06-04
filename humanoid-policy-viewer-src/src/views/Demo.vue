<template>
  <div id="mujoco-container"></div>
  <div class="global-alerts">
    <v-alert
      v-if="isSafari"
      v-model="showSafariAlert"
      type="warning"
      variant="flat"
      density="compact"
      closable
      class="safari-alert"
    >
      Safari has lower memory limits, which can cause WASM to crash.
    </v-alert>
  </div>
  <!-- Mobile UI -->
  <template v-if="isSmallScreen && state === 1">
    <!-- Joystick (bottom-left) -->
    <div class="mobile-joystick"
      @touchstart.prevent="onJoystickStart"
      @touchmove.prevent="onJoystickMove"
      @touchend.prevent="onJoystickEnd"
      @touchcancel.prevent="onJoystickEnd"
    >
      <div class="joystick-base">
        <div class="joystick-knob" :style="joystickKnobStyle"></div>
      </div>
    </div>
    <!-- Turn buttons (bottom-right) -->
    <div class="mobile-turn-buttons">
      <button class="mobile-turn-btn"
        @touchstart.prevent="startTurn(1)" @touchend.prevent="stopTurn()" @touchcancel.prevent="stopTurn()">&#x21B6;</button>
      <button class="mobile-turn-btn"
        @touchstart.prevent="startTurn(-1)" @touchend.prevent="stopTurn()" @touchcancel.prevent="stopTurn()">&#x21B7;</button>
    </div>
    <!-- Slide-up drawer toggle -->
    <button class="mobile-drawer-toggle" @click="mobileDrawerOpen = !mobileDrawerOpen">
      <span class="mobile-drawer-chevron" :class="{ open: mobileDrawerOpen }">&#x25B2;</span>
      <span>Controls</span>
    </button>
    <!-- Slide-up drawer -->
    <transition name="drawer-slide">
      <div v-if="mobileDrawerOpen" class="mobile-drawer">
        <div class="mobile-drawer-content">
          <div class="mobile-drawer-section">
            <div class="mobile-drawer-label">Task</div>
            <div class="mobile-task-toggle">
              <button
                v-for="option in plannerTaskOptions"
                :key="'m-task-'+option.value"
                class="mobile-task-btn"
                :class="{ active: plannerTask === option.value }"
                @click="setPlannerTask(option.value)"
              >
                {{ option.label }}
              </button>
            </div>
            <div class="mobile-drawer-label">Box Start</div>
            <div class="mobile-obj-grid">
              <span class="mobile-obj-grid-label">Pos</span>
              <input v-model.number="boxStart.x" type="number" min="-5" max="5" step="0.05" class="mobile-obj-input" placeholder="X"/>
              <input v-model.number="boxStart.y" type="number" min="-5" max="5" step="0.05" class="mobile-obj-input" placeholder="Y"/>
              <input v-model.number="boxStart.z" type="number" min="0.15" max="0.8" step="0.05" class="mobile-obj-input" :disabled="isPushTask" placeholder="Z"/>
            </div>
            <div class="mobile-drawer-label" style="margin-top:8px">Goal</div>
            <div class="mobile-obj-grid">
              <span class="mobile-obj-grid-label">Pos</span>
              <input v-model.number="goalPos.x" type="number" min="-5" max="5" step="0.05" class="mobile-obj-input" placeholder="X"/>
              <input v-model.number="goalPos.y" type="number" min="-5" max="5" step="0.05" class="mobile-obj-input" placeholder="Y"/>
              <input v-model.number="goalPos.z" type="number" min="0.15" max="0.8" step="0.05" class="mobile-obj-input" :disabled="isPushTask" placeholder="Z"/>
            </div>
            <div class="mobile-drawer-row" style="margin-top:8px">
              <button class="mobile-task-btn" @click="applyPlannerConfig()">Apply</button>
              <button class="mobile-task-btn mobile-replan-btn" :class="{ active: autoReplanEnabled }" @click="toggleAutoReplan()">
                {{ autoReplanEnabled ? 'Replan On' : 'Replan Off' }}
              </button>
              <button class="mobile-task-btn mobile-reset-btn" @click="reset()">Reset</button>
              <button class="mobile-task-btn mobile-reset-btn" @click="randomReset()">Random Reset</button>
            </div>
          </div>
          <div v-if="isInteractionMode" class="mobile-drawer-section">
            <div class="mobile-drawer-label">Upload Object</div>
            <div class="mobile-drawer-row">
              <input ref="mobileMeshFileInput" type="file" accept=".obj,.stl" style="display:none" @change="onMeshFileSelected" />
              <button class="mobile-task-btn" :disabled="objComputing" @click="$refs.mobileMeshFileInput.click()">Upload .obj/.stl</button>
            </div>
            <template v-if="userObjects.length > 0">
              <div class="mobile-drawer-label" style="margin-top:8px;cursor:pointer;display:flex;align-items:center;gap:4px" @click="mobileObjectListOpen = !mobileObjectListOpen">
                <span style="font-size:0.6rem">{{ mobileObjectListOpen ? '&#x25BC;' : '&#x25B6;' }}</span>
                Objects ({{ userObjects.length }})
              </div>
              <div v-if="mobileObjectListOpen">
                <div v-for="obj in userObjects" :key="'m-'+obj.name" class="mobile-obj-item-card">
                  <div class="mobile-obj-header" @click="obj.expanded = !obj.expanded">
                    <span class="mobile-obj-name">{{ obj.label }} <span v-if="obj.confirmed" style="color:#4caf50">&#x2713;</span></span>
                    <span style="font-size:0.55rem;color:#94a3b8">{{ obj.expanded ? '&#x25BC;' : '&#x25B6;' }}</span>
                  </div>
                  <div v-if="obj.expanded" class="mobile-obj-controls">
                    <div class="mobile-obj-grid">
                      <span class="mobile-obj-grid-label">Pos</span>
                      <input type="number" step="0.1" :value="obj.pos[0].toFixed(2)" class="mobile-obj-input" :disabled="obj.confirmed" @change="onObjPosChange(obj.name, 0, $event)" placeholder="X"/>
                      <input type="number" step="0.1" :value="obj.pos[1].toFixed(2)" class="mobile-obj-input" :disabled="obj.confirmed" @change="onObjPosChange(obj.name, 1, $event)" placeholder="Y"/>
                      <input type="number" step="0.1" :value="obj.pos[2].toFixed(2)" class="mobile-obj-input" :disabled="obj.confirmed" @change="onObjPosChange(obj.name, 2, $event)" placeholder="Z"/>
                    </div>
                    <div class="mobile-obj-grid">
                      <span class="mobile-obj-grid-label">Rot</span>
                      <input type="number" step="5" :value="obj.euler[0].toFixed(1)" class="mobile-obj-input" :disabled="obj.confirmed" @change="onObjEulerChange(obj.name, 0, $event)" placeholder="R"/>
                      <input type="number" step="5" :value="obj.euler[1].toFixed(1)" class="mobile-obj-input" :disabled="obj.confirmed" @change="onObjEulerChange(obj.name, 1, $event)" placeholder="P"/>
                      <input type="number" step="5" :value="obj.euler[2].toFixed(1)" class="mobile-obj-input" :disabled="obj.confirmed" @change="onObjEulerChange(obj.name, 2, $event)" placeholder="Y"/>
                    </div>
                    <div class="mobile-obj-slider-row">
                      <span class="mobile-obj-slider-label">Scale {{ obj.scale.toFixed(2) }}</span>
                      <input type="range" min="0.1" max="3.0" step="0.05" :value="obj.scale" class="mobile-obj-range" :disabled="obj.confirmed" @input="onObjScaleChange(obj.name, $event.target.value)"/>
                    </div>
                    <div class="mobile-obj-slider-row">
                      <span class="mobile-obj-slider-label">Mass {{ obj.mass.toFixed(1) }}kg</span>
                      <input type="range" min="0.5" max="20.0" step="0.1" :value="obj.mass" class="mobile-obj-range" :disabled="obj.confirmed" @input="onObjMassChange(obj.name, $event.target.value)"/>
                    </div>
                    <div class="mobile-obj-slider-row">
                      <span class="mobile-obj-slider-label">Friction {{ obj.friction.toFixed(1) }}</span>
                      <input type="range" min="0.1" max="3.0" step="0.1" :value="obj.friction" class="mobile-obj-range" :disabled="obj.confirmed" @input="onObjFrictionChange(obj.name, $event.target.value)"/>
                    </div>
                    <button v-if="!obj.confirmed" class="mobile-task-btn mobile-confirm-btn" style="width:100%;margin-top:6px" :disabled="obj.confirming||objComputing" @click="confirmUserObj(obj.name)">
                      {{ obj.confirming ? 'Computing SDF...' : 'Add to Simulation' }}
                    </button>
                  </div>
                </div>
              </div>
            </template>
          </div>
        </div>
      </div>
    </transition>
  </template>
  <div v-if="!isSmallScreen" class="controls">
    <v-card class="controls-card" elevation="0">
      <div class="controls-titlebar">
        <div class="controls-title-copy">
          <span class="controls-eyebrow">MuJoCo WASM + ONNX</span>
          <h1>OmniContact</h1>
        </div>
        <div class="viewer-state" :class="{ ready: state === 1, error: state < 0, loading: state === 0 }">
          <span class="viewer-state-dot"></span>
          <span>{{ viewerStateLabel }}</span>
        </div>
      </div>
      <v-card-text class="py-0 controls-body">

        <section class="control-section control-section-primary">
          <button class="section-header" type="button" @click="sectionTask = !sectionTask">
            <span class="section-title-group">
              <v-icon size="16">mdi-map-marker-path</v-icon>
              <span class="status-name">Planner</span>
            </span>
            <span class="section-meta" v-if="!sectionTask">box / goal</span>
            <v-icon size="16" class="section-chevron">{{ sectionTask ? 'mdi-chevron-up' : 'mdi-chevron-down' }}</v-icon>
          </button>
          <div v-if="sectionTask" class="planner-controls">
            <div class="field-label-row">
              <span>Task</span>
              <span>{{ plannerTaskOptions.find((option) => option.value === plannerTask)?.label }}</span>
            </div>
            <v-btn-toggle
              v-model="plannerTask"
              mandatory
              density="compact"
              variant="tonal"
              class="planner-task-toggle"
            >
              <v-btn
              v-for="option in plannerTaskOptions"
              :key="'task-'+option.value"
              size="small"
              :value="option.value"
              @click="setPlannerTask(option.value)"
            >
                {{ option.label }}
              </v-btn>
            </v-btn-toggle>
            <div class="pose-card-grid">
              <div class="pose-card">
                <div class="pose-card-title">
                  <v-icon size="15">mdi-cube-outline</v-icon>
                  <span>Box Start</span>
                </div>
                <div class="axis-grid">
                  <label>X</label>
                  <label>Y</label>
                  <label>Z</label>
                  <input v-model.number="boxStart.x" type="number" min="-5" max="5" step="0.05" class="pos-input" placeholder="X"/>
                  <input v-model.number="boxStart.y" type="number" min="-5" max="5" step="0.05" class="pos-input" placeholder="Y"/>
                  <input v-model.number="boxStart.z" type="number" min="0.15" max="0.8" step="0.05" class="pos-input" :disabled="isPushTask" placeholder="Z"/>
                </div>
              </div>
              <div class="pose-card">
                <div class="pose-card-title">
                  <v-icon size="15">mdi-crosshairs-gps</v-icon>
                  <span>Goal</span>
                </div>
                <div class="axis-grid">
                  <label>X</label>
                  <label>Y</label>
                  <label>Z</label>
                  <input v-model.number="goalPos.x" type="number" min="-5" max="5" step="0.05" class="pos-input" placeholder="X"/>
                  <input v-model.number="goalPos.y" type="number" min="-5" max="5" step="0.05" class="pos-input" placeholder="Y"/>
                  <input v-model.number="goalPos.z" type="number" min="0.15" max="0.8" step="0.05" class="pos-input" :disabled="isPushTask" placeholder="Z"/>
                </div>
              </div>
            </div>
            <div class="task-buttons">
              <v-btn size="small" variant="flat" color="primary" :disabled="state !== 1" prepend-icon="mdi-check" @click="applyPlannerConfig">Apply</v-btn>
              <v-btn size="small" :variant="autoReplanEnabled ? 'flat' : 'tonal'" color="warning" :disabled="state !== 1" prepend-icon="mdi-autorenew" @click="toggleAutoReplan">
                {{ autoReplanEnabled ? 'Replan On' : 'Replan Off' }}
              </v-btn>
              <v-btn size="small" variant="tonal" :disabled="state !== 1" prepend-icon="mdi-restart" @click="reset">Reset</v-btn>
              <v-btn size="small" variant="tonal" :disabled="state !== 1" prepend-icon="mdi-shuffle-variant" @click="randomReset">Random</v-btn>
            </div>
          </div>
        </section>

        <template v-if="isInteractionMode">
        <section class="control-section">
        <button class="section-header" type="button" @click="sectionObject = !sectionObject">
          <span class="section-title-group">
            <v-icon size="16">mdi-shape-plus</v-icon>
            <span class="status-name">Objects</span>
          </span>
          <span class="section-meta" v-if="!sectionObject && userObjects.length">{{ userObjects.length }} obj</span>
          <v-icon size="16" class="section-chevron">{{ sectionObject ? 'mdi-chevron-up' : 'mdi-chevron-down' }}</v-icon>
        </button>
        <div v-if="sectionObject" class="obj-upload">
          <input
            ref="meshFileInput"
            type="file"
            accept=".obj,.stl"
            style="display:none"
            @change="onMeshFileSelected"
          />
          <v-btn
            size="small"
            variant="tonal"
            :disabled="state !== 1 || objComputing"
            prepend-icon="mdi-upload"
            @click="$refs.meshFileInput.click()"
          >
            Upload Mesh
          </v-btn>
        </div>
        <div v-if="sectionObject && userObjects.length > 0" class="mt-2">
          <button class="section-header subsection-header" type="button" @click="sectionObjectList = !sectionObjectList">
            <span class="section-title-group">
              <v-icon size="14">mdi-format-list-bulleted</v-icon>
              <span class="status-name text-caption">Objects ({{ userObjects.length }})</span>
            </span>
            <v-icon size="14" class="section-chevron">{{ sectionObjectList ? 'mdi-chevron-up' : 'mdi-chevron-down' }}</v-icon>
          </button>
          <div v-if="sectionObjectList" class="user-objects-list mt-1">
          <div v-for="obj in userObjects" :key="obj.name" class="user-object-item">
            <div class="user-object-header">
              <v-btn
                icon
                size="x-small"
                variant="text"
                @click="obj.expanded = !obj.expanded"
              >
                <v-icon size="14">{{ obj.expanded ? 'mdi-chevron-up' : 'mdi-chevron-down' }}</v-icon>
              </v-btn>
              <span class="text-caption text-truncate" :title="obj.name" style="flex:1">
                {{ obj.label }}
                <span v-if="obj.confirmed" class="text-caption" style="color: #4caf50;">(physics)</span>
              </span>
              <v-btn
                icon
                size="x-small"
                variant="text"
                color="error"
                :disabled="obj.confirming"
                @click="removeUserObj(obj.name)"
              >
                <v-icon size="14">mdi-close</v-icon>
              </v-btn>
            </div>
            <div v-if="obj.expanded" class="user-object-controls">
              <div class="user-object-grid">
                <label class="text-caption grid-label">Pos</label>
                <input type="number" step="0.1" :value="obj.pos[0].toFixed(2)" class="pos-input" :disabled="obj.confirmed" @change="onObjPosChange(obj.name, 0, $event)" placeholder="X"/>
                <input type="number" step="0.1" :value="obj.pos[1].toFixed(2)" class="pos-input" :disabled="obj.confirmed" @change="onObjPosChange(obj.name, 1, $event)" placeholder="Y"/>
                <input type="number" step="0.1" :value="obj.pos[2].toFixed(2)" class="pos-input" :disabled="obj.confirmed" @change="onObjPosChange(obj.name, 2, $event)" placeholder="Z"/>
                <label class="text-caption grid-label">Rot</label>
                <input type="number" step="5" :value="obj.euler[0].toFixed(1)" class="pos-input" :disabled="obj.confirmed" @change="onObjEulerChange(obj.name, 0, $event)" placeholder="R"/>
                <input type="number" step="5" :value="obj.euler[1].toFixed(1)" class="pos-input" :disabled="obj.confirmed" @change="onObjEulerChange(obj.name, 1, $event)" placeholder="P"/>
                <input type="number" step="5" :value="obj.euler[2].toFixed(1)" class="pos-input" :disabled="obj.confirmed" @change="onObjEulerChange(obj.name, 2, $event)" placeholder="Y"/>
              </div>
              <div class="user-object-scale">
                <label class="text-caption">Scale {{ obj.scale.toFixed(2) }}</label>
                <v-slider :model-value="obj.scale" min="0.1" max="3.0" step="0.05" density="compact" hide-details :disabled="obj.confirmed" @update:modelValue="onObjScaleChange(obj.name, $event)"></v-slider>
              </div>
              <div class="user-object-scale">
                <label class="text-caption">Mass {{ obj.mass.toFixed(1) }}kg</label>
                <v-slider :model-value="obj.mass" min="0.5" max="20.0" step="0.1" density="compact" hide-details :disabled="obj.confirmed" @update:modelValue="onObjMassChange(obj.name, $event)"></v-slider>
              </div>
              <div class="user-object-scale">
                <label class="text-caption">Friction {{ obj.friction.toFixed(1) }}</label>
                <v-slider :model-value="obj.friction" min="0.1" max="3.0" step="0.1" density="compact" hide-details :disabled="obj.confirmed" @update:modelValue="onObjFrictionChange(obj.name, $event)"></v-slider>
              </div>
              <v-btn
                v-if="!obj.confirmed"
                size="small"
                variant="tonal"
                color="success"
                block
                class="mt-1"
                :loading="obj.confirming"
                :disabled="obj.confirming || objComputing"
                @click="confirmUserObj(obj.name)"
              >
                {{ obj.confirming ? 'Computing SDF & Adding...' : 'Add to Simulation' }}
              </v-btn>
              <v-progress-linear
                v-if="obj.confirming && objProgress > 0"
                :model-value="objProgress"
                color="success"
                height="4"
                class="mt-1"
              ></v-progress-linear>
            </div>
          </div>
          </div>
        </div>

        <v-tooltip location="bottom" text="Red arrows show the distance field gradient the policy uses to sense nearby objects.">
          <template v-slot:activator="{ props }">
            <v-checkbox
              v-if="sectionObject"
              v-bind="props"
              v-model="sdfVisEnabled"
              label="Visualize SDF"
              density="compact"
              hide-details
              class="mt-1 sdf-vis-checkbox"
              :disabled="state !== 1"
              @update:modelValue="onSdfVisToggle"
            ></v-checkbox>
          </template>
        </v-tooltip>
        </section>
        </template>

        <section class="control-section">
        <button class="section-header" type="button" @click="sectionSettings = !sectionSettings">
          <span class="section-title-group">
            <v-icon size="16">mdi-tune-variant</v-icon>
            <span class="status-name">Settings</span>
          </span>
          <span class="section-meta" v-if="!sectionSettings">{{ renderScaleLabel }}</span>
          <v-icon size="16" class="section-chevron">{{ sectionSettings ? 'mdi-chevron-up' : 'mdi-chevron-down' }}</v-icon>
        </button>
        <template v-if="sectionSettings">
        <div class="settings-grid">
        <div class="status-legend follow-controls">
          <span class="status-name">Camera follow</span>
          <v-btn
            size="x-small"
            variant="tonal"
            color="primary"
            :disabled="state !== 1"
            @click="toggleCameraFollow"
          >
            {{ cameraFollowEnabled ? 'On' : 'Off' }}
          </v-btn>
        </div>
        <div class="status-legend">
          <span class="status-name">Sim Freq</span>
          <span class="text-caption">{{ simStepLabel }}</span>
        </div>
        </div>
        <div class="field-label-row slider-label-row">
          <span>Render scale</span>
          <span>{{ renderScaleLabel }}</span>
        </div>
        <v-slider
          v-model="renderScale"
          min="0.5"
          max="2.0"
          step="0.1"
          density="compact"
          hide-details
          @update:modelValue="onRenderScaleChange"
        ></v-slider>
        </template>
        </section>
      </v-card-text>
    </v-card>
  </div>
  <v-dialog :model-value="state === 0" persistent max-width="600px" scrollable>
    <v-card title="Loading Simulation Environment">
      <v-card-text>
        <v-progress-linear indeterminate color="primary"></v-progress-linear>
        Loading MuJoCo and ONNX policy, please wait
      </v-card-text>
    </v-card>
  </v-dialog>
  <v-dialog :model-value="state < 0" persistent max-width="600px" scrollable>
    <v-card title="Simulation Environment Loading Error">
      <v-card-text>
        <span v-if="state === -1">
          Unexpected runtime error, please refresh the page.<br />
          {{ extra_error_message }}
        </span>
        <span v-else-if="state === -2">
          Your browser does not support WebAssembly. Please use a recent version of Chrome, Edge, or Firefox.
        </span>
      </v-card-text>
    </v-card>
  </v-dialog>
</template>

<script>
import { MuJoCoDemo } from '@/simulation/main.js';
import loadMujoco from 'mujoco-js';

export default {
  name: 'DemoPage',
  data: () => ({
    state: 0, // 0: loading, 1: running, -1: JS error, -2: wasm unsupported
    extra_error_message: '',
    keydown_listener: null,
    currentMotion: null,
    availableMotions: [],
    trackingState: {
      available: false,
      currentName: 'default',
      currentDone: true,
      refIdx: 0,
      refLen: 0,
      transitionLen: 0,
      motionLen: 0,
      inTransition: false,
      isDefault: true
    },
    trackingTimer: null,
    plannerTask: 'carrybox',
    plannerTaskOptions: [
      { value: 'carrybox', label: 'Carry' },
      { value: 'pushbox', label: 'Push' },
      { value: 'pushbox-old', label: 'Push 2' }
    ],
    carryBoxDefaultHeight: 0.16,
    carryGoalDefaultHeight: 0.55,
    pushBoxHalfHeight: 0.26,
    boxStart: { x: 1.0, y: 0.0, z: 0.16 },
    goalPos: { x: 1.0, y: 1.0, z: 0.55 },
    // OBJ upload
    objComputing: false,
    objProgress: 0,
    userObjects: [],   // [{ name, label, pos: [x,y,z], euler: [rx,ry,rz] (deg), scale, confirmed, expanded }]
    // Collapsible panel sections
    sectionMovement: true,
    sectionTask: true,
    sectionObject: true,
    sectionSettings: false,
    cameraFollowEnabled: true,
    autoReplanEnabled: false,
    // Mobile joystick state
    joystickX: 0,
    joystickY: 0,
    joystickActive: false,
    _joystickRafId: null,
    _turnDir: 0,
    _turnRafId: null,
    sdfVisEnabled: true,
    mobileDrawerOpen: false,
    mobileObjectListOpen: true,
    sectionObjectList: true,
    sdfResolution: 32,
    renderScale: 2.0,
    simStepHz: 0,
    isInteractionMode: false,
    isSmallScreen: false,
    showSmallScreenAlert: true,
    isSafari: false,
    showSafariAlert: true,
    resize_listener: null
  }),
  computed: {
    viewerStateLabel() {
      if (this.state === 1) {
        return 'Ready';
      }
      if (this.state < 0) {
        return 'Error';
      }
      return 'Loading';
    },
    isPushTask() {
      return this.plannerTask === 'pushbox' || this.plannerTask === 'pushbox-old';
    },
    renderScaleLabel() {
      return `${this.renderScale.toFixed(2)}x`;
    },
    simStepLabel() {
      if (!this.simStepHz || !Number.isFinite(this.simStepHz)) {
        return '—';
      }
      return `${this.simStepHz.toFixed(1)} Hz`;
    },
    joystickKnobStyle() {
      return {
        transform: `translate(${this.joystickX * 30}px, ${this.joystickY * -30}px)`
      };
    }
  },
  watch: {
    plannerTask(newTask, oldTask) {
      this.syncTaskHeights();
      if (oldTask !== undefined && newTask !== oldTask && this.state === 1) {
        void this.applyPlannerConfig({ resetSimulation: true });
      }
    }
  },
  methods: {
    setPlannerTask(task) {
      if (this.plannerTask === task) {
        return;
      }
      this.plannerTask = task;
    },
    syncTaskHeights() {
      if (this.isPushTask) {
        this.syncPushHeights();
        return;
      }
      this.boxStart.z = this.carryBoxDefaultHeight;
      this.goalPos.z = this.carryGoalDefaultHeight;
    },
    syncPushHeights() {
      if (!this.isPushTask) {
        return;
      }
      this.boxStart.z = this.pushBoxHalfHeight;
      this.goalPos.z = this.pushBoxHalfHeight;
    },
    detectSafari() {
      const ua = navigator.userAgent;
      return /Safari\//.test(ua)
        && !/Chrome\//.test(ua)
        && !/Chromium\//.test(ua)
        && !/Edg\//.test(ua)
        && !/OPR\//.test(ua)
        && !/SamsungBrowser\//.test(ua)
        && !/CriOS\//.test(ua)
        && !/FxiOS\//.test(ua);
    },
    updateScreenState() {
      const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
      const isSmallViewport = window.innerWidth < 500 || window.innerHeight < 700;
      this.isSmallScreen = isTouchDevice && isSmallViewport;
    },
    async init() {
      if (typeof WebAssembly !== 'object' || typeof WebAssembly.instantiate !== 'function') {
        this.state = -2;
        return;
      }

      try {
        const mujoco = await loadMujoco();
        this.demo = new MuJoCoDemo(mujoco);
        this.demo.setFollowEnabled?.(this.cameraFollowEnabled);
        await this.demo.init();
        this.demo.main_loop();
        this.demo.params.paused = false;
        this.isInteractionMode = Boolean(this.demo.isInteractionMode);
        this.startTrackingPoll();
        this.renderScale = this.demo.renderScale ?? this.renderScale;
        this.state = 1;
        // Signal parent page that the viewer is ready (dismisses loading overlay)
        try { window.parent.postMessage({ type: 'omnicontact-ready' }, '*'); } catch (e) {}
      } catch (error) {
        this.state = -1;
        this.extra_error_message = error.toString();
        console.error(error);
      }
    },
    toggleCameraFollow() {
      this.cameraFollowEnabled = !this.cameraFollowEnabled;
      if (this.demo?.setFollowEnabled) {
        this.demo.setFollowEnabled(this.cameraFollowEnabled);
      }
    },
    onSdfVisToggle(value) {
      if (this.demo?.setSdfVisEnabled) {
        this.demo.setSdfVisEnabled(value);
      }
    },
    clampPlannerValue(value, min, max) {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) {
        return min;
      }
      return Math.min(max, Math.max(min, numeric));
    },
    randomPlannerValue(min, max) {
      return Number((min + Math.random() * (max - min)).toFixed(2));
    },
    buildPlannerConfig() {
      this.boxStart.x = this.clampPlannerValue(this.boxStart.x, -5, 5);
      this.boxStart.y = this.clampPlannerValue(this.boxStart.y, -5, 5);
      this.boxStart.z = this.clampPlannerValue(this.boxStart.z, 0.15, 0.8);
      this.goalPos.x = this.clampPlannerValue(this.goalPos.x, -5, 5);
      this.goalPos.y = this.clampPlannerValue(this.goalPos.y, -5, 5);
      this.goalPos.z = this.clampPlannerValue(this.goalPos.z, 0.15, 0.8);
      this.syncPushHeights();
      const boxStartPos = [
        Number(this.boxStart.x),
        Number(this.boxStart.y),
        Number(this.boxStart.z)
      ];
      const goalPos = [
        Number(this.goalPos.x),
        Number(this.goalPos.y),
        Number(this.goalPos.z)
      ];
      if (boxStartPos.some((v) => !Number.isFinite(v)) || goalPos.some((v) => !Number.isFinite(v))) {
        return null;
      }
      return { boxStartPos, goalPos, task: this.plannerTask };
    },
    async applyPlannerConfig(options = {}) {
      const config = this.buildPlannerConfig();
      if (!config) {
        return;
      }
      await this.demo?.applyCarryBoxPlannerConfig?.({
        ...config,
        resetSimulation: Boolean(options.resetSimulation)
      });
    },
    toggleAutoReplan() {
      const config = this.buildPlannerConfig();
      if (!config) {
        return;
      }
      this.autoReplanEnabled = !this.autoReplanEnabled;
      this.demo?.updateCarryBoxReplanGoal?.({ goalPos: config.goalPos, task: config.task });
      this.demo?.setAutoReplanEnabled?.(this.autoReplanEnabled);
    },
    randomReset() {
      this.boxStart = {
        x: this.randomPlannerValue(-5, 5),
        y: this.randomPlannerValue(-5, 5),
        z: this.isPushTask ? this.pushBoxHalfHeight : this.randomPlannerValue(0.15, 0.8)
      };
      this.goalPos = {
        x: this.randomPlannerValue(-5, 5),
        y: this.randomPlannerValue(-5, 5),
        z: this.isPushTask ? this.pushBoxHalfHeight : this.randomPlannerValue(0.15, 0.8)
      };
      void this.applyPlannerConfig({ resetSimulation: true });
    },
    moveTargetRoot(dx, dy, dz) {
      const runner = this.demo?.interactionRunner;
      if (!runner) return;
      // Apply movement in the target root's frame
      const q = runner.targetRootQuat;
      const w = q[0], qx = q[1], qy = q[2], qz = q[3];
      // Rotate direction by target quat
      const tx = 2 * (qy * dz - qz * dy);
      const ty = 2 * (qz * dx - qx * dz);
      const tz = 2 * (qx * dy - qy * dx);
      runner.targetRootPos[0] += dx + w * tx + (qy * tz - qz * ty);
      runner.targetRootPos[1] += dy + w * ty + (qz * tx - qx * tz);
      runner.targetRootPos[2] += dz + w * tz + (qx * ty - qy * tx);
    },
    rotateTargetRoot(yaw) {
      const runner = this.demo?.interactionRunner;
      if (!runner) return;
      const halfYaw = yaw * 0.5;
      const cz = Math.cos(halfYaw);
      const sz = Math.sin(halfYaw);
      // Quaternion for yaw rotation: [cos(y/2), 0, 0, sin(y/2)]
      const rw = cz, rx = 0, ry = 0, rz = sz;
      const q = runner.targetRootQuat;
      const aw = q[0], ax = q[1], ay = q[2], az = q[3];
      // q_new = q_yaw * q_current
      runner.targetRootQuat[0] = rw * aw - rx * ax - ry * ay - rz * az;
      runner.targetRootQuat[1] = rw * ax + rx * aw + ry * az - rz * ay;
      runner.targetRootQuat[2] = rw * ay - rx * az + ry * aw + rz * ax;
      runner.targetRootQuat[3] = rw * az + rx * ay - ry * ax + rz * aw;
    },
    // --- Mobile joystick ---
    onJoystickStart(e) {
      this.joystickActive = true;
      this._joystickOrigin = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      this._joystickLastTime = performance.now();
      if (!this._joystickRafId) this._joystickRafId = requestAnimationFrame(this._joystickLoop.bind(this));
    },
    onJoystickMove(e) {
      if (!this.joystickActive || !this._joystickOrigin) return;
      const dx = e.touches[0].clientX - this._joystickOrigin.x;
      const dy = e.touches[0].clientY - this._joystickOrigin.y;
      const maxR = 30;
      const len = Math.sqrt(dx * dx + dy * dy);
      const clamp = Math.min(len, maxR) / maxR;
      if (len > 0) {
        this.joystickX = (dx / len) * clamp;
        this.joystickY = -(dy / len) * clamp; // invert Y: up = positive
      }
    },
    onJoystickEnd() {
      this.joystickActive = false;
      this.joystickX = 0;
      this.joystickY = 0;
      this._joystickRafId = null;
    },
    _joystickLoop() {
      if (!this.joystickActive) { this._joystickRafId = null; return; }
      const now = performance.now();
      let dt = (now - this._joystickLastTime) / 1000;
      if (dt > 0.05) dt = 0.05;
      this._joystickLastTime = now;
      const SPEED = 1.5;
      if (Math.abs(this.joystickY) > 0.05) this.moveTargetRoot(this.joystickY * SPEED * dt, 0, 0);
      if (Math.abs(this.joystickX) > 0.05) this.moveTargetRoot(0, -this.joystickX * SPEED * dt, 0);
      this._joystickRafId = requestAnimationFrame(this._joystickLoop.bind(this));
    },
    startTurn(dir) {
      this._turnDir = dir;
      this._turnLastTime = performance.now();
      if (!this._turnRafId) this._turnRafId = requestAnimationFrame(this._turnLoop.bind(this));
    },
    stopTurn() {
      this._turnDir = 0;
      this._turnRafId = null;
    },
    _turnLoop() {
      if (this._turnDir === 0) { this._turnRafId = null; return; }
      const now = performance.now();
      let dt = (now - this._turnLastTime) / 1000;
      if (dt > 0.05) dt = 0.05;
      this._turnLastTime = now;
      this.rotateTargetRoot(this._turnDir * 2.0 * dt);
      this._turnRafId = requestAnimationFrame(this._turnLoop.bind(this));
    },
    async onMeshFileSelected(event) {
      const file = event.target.files?.[0];
      if (!file || !this.demo) return;
      event.target.value = ''; // allow re-upload of same file

      const ext = file.name.split('.').pop().toLowerCase();
      const label = file.name.replace(/\.(obj|stl)$/i, '');

      try {
        let result;
        if (ext === 'stl') {
          const buffer = await file.arrayBuffer();
          result = this.demo.addUserObjectFromStl(buffer, {
            name: undefined,
            position: [0.7, 0.7, 0.2],
          });
        } else {
          const text = await file.text();
          result = this.demo.addUserObject(text, {
            name: undefined,
            position: [0.7, 0.7, 0.2],
          });
        }

        this.userObjects.push({
          name: result.name,
          label,
          pos: [0.7, 0.7, 0.2],
          euler: [0, 0, 0],   // degrees
          scale: 1.0,
          mass: 1.0,
          friction: 1.0,
          confirmed: false,
          confirming: false,
          expanded: true,
        });
      } catch (err) {
        console.error('Failed to load mesh:', err);
        alert('Failed to load mesh file: ' + err.message);
      }
    },
    onObjPosChange(name, axis, event) {
      const val = parseFloat(event.target.value);
      if (!Number.isFinite(val)) return;
      const obj = this.userObjects.find(o => o.name === name);
      if (!obj || obj.confirmed) return;
      obj.pos[axis] = val;
      const eulerRad = obj.euler.map(d => d * Math.PI / 180);
      this.demo?.setUserObjectPoseAndRotation(name, [...obj.pos], eulerRad);
    },
    onObjEulerChange(name, axis, event) {
      const val = parseFloat(event.target.value);
      if (!Number.isFinite(val)) return;
      const obj = this.userObjects.find(o => o.name === name);
      if (!obj || obj.confirmed) return;
      obj.euler[axis] = val;
      const eulerRad = obj.euler.map(d => d * Math.PI / 180);
      this.demo?.setUserObjectPoseAndRotation(name, [...obj.pos], eulerRad);
    },
    onObjScaleChange(name, value) {
      const s = parseFloat(value);
      if (!Number.isFinite(s) || s <= 0) return;
      const obj = this.userObjects.find(o => o.name === name);
      if (!obj || obj.confirmed) return;
      obj.scale = s;
      this.demo?.setUserObjectScale(name, s);
    },
    onObjMassChange(name, value) {
      const obj = this.userObjects.find(o => o.name === name);
      if (!obj || obj.confirmed) return;
      obj.mass = parseFloat(value);
    },
    onObjFrictionChange(name, value) {
      const obj = this.userObjects.find(o => o.name === name);
      if (!obj || obj.confirmed) return;
      obj.friction = parseFloat(value);
    },
    async confirmUserObj(name) {
      const obj = this.userObjects.find(o => o.name === name);
      if (!obj || obj.confirmed || obj.confirming) return;

      obj.confirming = true;
      this.objComputing = true;
      this.objProgress = 0;
      try {
        const eulerRad = obj.euler.map(d => d * Math.PI / 180);
        const progressCb = (done, total) => {
          this.objProgress = Math.round((done / total) * 100);
        };
        const success = await this.demo?.confirmUserObject(
          name, [...obj.pos], eulerRad, obj.scale, obj.mass, obj.friction, this.sdfResolution, progressCb
        );
        if (success) {
          obj.confirmed = true;
          obj.expanded = false;
        } else {
          alert('Failed to add object to simulation.');
        }
      } catch (err) {
        console.error('Confirm object error:', err);
        alert('Error adding object to simulation: ' + err.message);
      } finally {
        obj.confirming = false;
        this.objComputing = false;
        this.objProgress = 0;
      }
    },
    async removeUserObj(name) {
      const obj = this.userObjects.find(o => o.name === name);
      if (!obj) return;
      // Remove from UI list immediately
      this.userObjects = this.userObjects.filter(o => o.name !== name);
      // Remove from simulation (async for confirmed objects — reloads scene)
      await this.demo?.removeUserObject(name);
    },
    reset() {
      if (!this.demo) {
        return;
      }
      if (this.isInteractionMode || !this.demo.applyCarryBoxPlannerConfig) {
        this.demo.resetSimulation();
        return;
      }
      void this.applyPlannerConfig({ resetSimulation: true });
    },
    startTrackingPoll() {
      this.stopTrackingPoll();
      this.updatePerformanceStats();
      this.trackingTimer = setInterval(() => {
        this.updatePerformanceStats();
      }, 33);
    },
    stopTrackingPoll() {
      if (this.trackingTimer) {
        clearInterval(this.trackingTimer);
        this.trackingTimer = null;
      }
    },
    updatePerformanceStats() {
      if (!this.demo) {
        this.simStepHz = 0;
        return;
      }
      this.simStepHz = this.demo.getSimStepHz?.() ?? this.demo.simStepHz ?? 0;
    },
    onRenderScaleChange(value) {
      if (!this.demo) {
        return;
      }
      this.demo.setRenderScale(value);
    }
  },
  mounted() {
    this.isSafari = this.detectSafari();
    this.updateScreenState();
    this.resize_listener = () => {
      this.updateScreenState();
    };
    window.addEventListener('resize', this.resize_listener);
    this.init();

    // Resource cleanup listeners
    window.addEventListener('pagehide', this._onPageHide = () => {
      this.demo?.dispose();
    });
    window.addEventListener('message', this._onMessage = (e) => {
      if (e.data?.type === 'omnicontact-close') this.demo?.dispose();
    });

    const MOVE_SPEED = 1.5;  // units per second
    const TURN_SPEED = 2.0;  // radians per second
    const MOVEMENT_KEYS = new Set(['w', 's', 'a', 'd', 'q', 'e']);
    this._pressedKeys = new Set();
    this._movementRafId = null;
    this._movementLastTime = 0;

    const movementLoop = () => {
      if (this._pressedKeys.size === 0) {
        this._movementRafId = null;
        return;
      }
      const now = performance.now();
      let dt = (now - this._movementLastTime) / 1000;
      if (dt > 0.05) dt = 0.05;  // cap to avoid jumps after tab switch
      this._movementLastTime = now;

      if (this._pressedKeys.has('w')) this.moveTargetRoot(MOVE_SPEED * dt, 0, 0);
      if (this._pressedKeys.has('s')) this.moveTargetRoot(-MOVE_SPEED * dt, 0, 0);
      if (this._pressedKeys.has('a')) this.moveTargetRoot(0, MOVE_SPEED * dt, 0);
      if (this._pressedKeys.has('d')) this.moveTargetRoot(0, -MOVE_SPEED * dt, 0);
      if (this._pressedKeys.has('q')) this.rotateTargetRoot(TURN_SPEED * dt);
      if (this._pressedKeys.has('e')) this.rotateTargetRoot(-TURN_SPEED * dt);

      this._movementRafId = requestAnimationFrame(movementLoop);
    };

    this.keydown_listener = (event) => {
      if (event.code === 'Backspace') {
        this.reset();
        return;
      }
      const key = event.key.toLowerCase();
      // Movement keys — continuous via RAF loop
      if (MOVEMENT_KEYS.has(key) && !this._pressedKeys.has(key)) {
        this._pressedKeys.add(key);
        if (!this._movementRafId) {
          this._movementLastTime = performance.now();
          this._movementRafId = requestAnimationFrame(movementLoop);
        }
      }
    };
    this._keyup_listener = (event) => {
      this._pressedKeys.delete(event.key.toLowerCase());
    };
    document.addEventListener('keydown', this.keydown_listener);
    document.addEventListener('keyup', this._keyup_listener);
  },
  beforeUnmount() {
    this.stopTrackingPoll();
    document.removeEventListener('keydown', this.keydown_listener);
    document.removeEventListener('keyup', this._keyup_listener);
    if (this._movementRafId) {
      cancelAnimationFrame(this._movementRafId);
      this._movementRafId = null;
    }
    if (this.resize_listener) {
      window.removeEventListener('resize', this.resize_listener);
    }
    window.removeEventListener('pagehide', this._onPageHide);
    window.removeEventListener('message', this._onMessage);
    this.demo?.dispose();
  }
};
</script>

<style scoped>
.controls {
  position: fixed;
  top: clamp(12px, 1.2vw, 24px);
  left: clamp(12px, 1.2vw, 24px);
  width: clamp(320px, 23vw, 430px);
  z-index: 1000;
  font-size: clamp(0.72rem, 0.82vw, 0.95rem);
}

.global-alerts {
  position: fixed;
  top: 20px;
  left: 16px;
  right: 16px;
  max-width: 520px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 1200;
}

.small-screen-alert {
  width: 100%;
}

.safari-alert {
  width: 100%;
}

.controls-card {
  max-height: calc(100vh - clamp(24px, 2.4vw, 48px));
  overflow: hidden;
  border: 1px solid rgba(226, 232, 240, 0.16);
  border-radius: 8px;
  background:
    linear-gradient(180deg, rgba(18, 26, 38, 0.94), rgba(8, 13, 22, 0.9)),
    rgba(15, 23, 42, 0.92) !important;
  color: #e5edf6;
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.06);
  backdrop-filter: blur(14px);
}

.controls-titlebar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 15px 12px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.18);
  background: linear-gradient(135deg, rgba(30, 41, 59, 0.72), rgba(12, 18, 30, 0.2));
}

.controls-title-copy {
  min-width: 0;
}

.controls-eyebrow {
  display: block;
  color: #ffb0b0;
  font-size: 0.63rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  line-height: 1;
  text-transform: uppercase;
}

.controls-title-copy h1 {
  margin: 4px 0 0;
  color: #f8fafc;
  font-size: 1.15rem;
  font-weight: 850;
  line-height: 1.1;
}

.viewer-state {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  min-height: 28px;
  padding: 0 10px;
  border: 1px solid rgba(148, 163, 184, 0.26);
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.74);
  color: #cbd5e1;
  font-size: 0.72rem;
  font-weight: 700;
  white-space: nowrap;
}

.viewer-state-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #94a3b8;
  box-shadow: 0 0 0 3px rgba(148, 163, 184, 0.12);
}

.viewer-state.ready .viewer-state-dot {
  background: #22c55e;
  box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.16);
}

.viewer-state.error .viewer-state-dot {
  background: #ef4444;
  box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.18);
}

.viewer-state.loading .viewer-state-dot {
  background: #f59e0b;
  box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.16);
}

.controls-body {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: calc(100vh - clamp(92px, 8vw, 150px));
  overflow-y: auto;
  overscroll-behavior: contain;
  padding: 12px !important;
}

.control-section {
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 8px;
  background: rgba(15, 23, 42, 0.54);
  overflow: hidden;
}

.control-section-primary {
  border-color: rgba(238, 99, 99, 0.22);
  background: linear-gradient(180deg, rgba(20, 31, 44, 0.78), rgba(13, 20, 31, 0.64));
}

.motion-status {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.motion-groups {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 12px;
  max-height: 200px;
  overflow-y: auto;
}

.motion-group {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.motion-chip {
  text-transform: none;
  font-size: 0.7rem;
}

.task-buttons {
  display: grid;
  grid-template-columns: 1.05fr 1.05fr 0.85fr 0.85fr;
  gap: 6px;
  margin-top: 2px;
}

.task-buttons .v-btn {
  min-width: 0;
  font-size: 0.68rem !important;
  letter-spacing: 0;
  text-transform: none;
}

.planner-controls {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 0 12px 12px;
}

.field-label-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: #94a3b8;
  font-size: 0.67rem;
  font-weight: 800;
  letter-spacing: 0.04em;
  line-height: 1;
  text-transform: uppercase;
}

.field-label-row span:last-child {
  color: #e2e8f0;
  letter-spacing: 0;
  text-transform: none;
}

.planner-task-toggle {
  align-self: stretch;
  width: 100%;
  height: 34px;
  padding: 3px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 8px;
  background: rgba(2, 6, 23, 0.42);
}

.planner-task-toggle .v-btn {
  flex: 1 1 0;
  min-width: 0;
  height: 28px !important;
  border-radius: 6px !important;
  font-size: 0.68rem !important;
  letter-spacing: 0;
  text-transform: none;
  color: #cbd5e1 !important;
}

.planner-task-toggle .v-btn :deep(.v-btn__content) {
  color: #cbd5e1 !important;
}

.planner-task-toggle .v-btn.v-btn--active,
.planner-task-toggle .v-btn.v-btn--selected {
  background: rgba(238, 99, 99, 0.88) !important;
  color: #ffffff !important;
}

.planner-task-toggle .v-btn.v-btn--active :deep(.v-btn__content),
.planner-task-toggle .v-btn.v-btn--selected :deep(.v-btn__content) {
  color: #ffffff !important;
}

.pose-card-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
}

.pose-card {
  padding: 9px;
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 8px;
  background: rgba(2, 6, 23, 0.26);
}

.pose-card-title {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
  color: #dbeafe;
  font-size: 0.72rem;
  font-weight: 800;
}

.axis-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 5px;
}

.axis-grid label {
  color: #94a3b8;
  font-size: 0.6rem;
  font-weight: 800;
  line-height: 1;
  text-align: center;
}

.user-objects-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 180px;
  overflow-y: auto;
  padding: 0 12px 10px;
}

.user-object-item {
  border: 1px solid rgba(148, 163, 184, 0.14);
  background: rgba(2, 6, 23, 0.3);
  border-radius: 6px;
  padding: 6px 8px;
}

.user-object-header {
  display: flex;
  align-items: center;
  gap: 2px;
}

.user-object-controls {
  margin-top: 2px;
}

.user-object-grid {
  display: grid;
  grid-template-columns: auto 1fr 1fr 1fr;
  gap: 2px 4px;
  align-items: center;
  margin-top: 2px;
}

.grid-label {
  font-weight: 600;
  white-space: nowrap;
}

.pos-input {
  width: 100%;
  min-width: 0;
  height: 30px;
  padding: 0 6px;
  font-size: 0.72rem;
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 6px;
  text-align: center;
  background: rgba(15, 23, 42, 0.55);
  color: inherit;
}

.pos-input:focus {
  outline: 1px solid rgba(238, 99, 99, 0.76);
  border-color: rgba(238, 99, 99, 0.62);
}

.user-object-scale {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 2px;
}

.user-object-scale label {
  white-space: nowrap;
  min-width: clamp(36px, 3vw, 56px);
  font-size: clamp(0.6rem, 0.7vw, 0.85rem);
}

.user-object-scale .v-slider {
  flex: 1;
  padding: 0 !important;
}

.sdf-vis-checkbox {
  margin: 0 12px 10px !important;
}

.sdf-vis-checkbox :deep(.v-label) {
  font-size: 0.8rem;
}

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  width: 100%;
  min-height: 40px;
  padding: 0 12px;
  border: 0;
  border-radius: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
  font: inherit;
  text-align: left;
  user-select: none;
}

.section-header .v-icon {
  font-size: clamp(12px, 0.9vw, 18px) !important;
}

.section-header:hover {
  background: rgba(148, 163, 184, 0.07);
}

.section-chevron {
  flex-shrink: 0;
  color: #94a3b8;
}

.section-title-group {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.section-title-group .v-icon {
  color: #ffb0b0;
}

.section-meta {
  margin-left: auto;
  color: #94a3b8;
  font-size: 0.68rem;
  font-weight: 700;
}

.subsection-header {
  min-height: 32px;
  padding: 0 6px;
  border-radius: 6px;
}

.status-legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.status-name {
  font-weight: 800;
}

.policy-file {
  display: block;
  margin-top: 4px;
}

.obj-upload {
  padding: 0 12px 10px;
}

.obj-upload .v-btn {
  width: 100%;
  justify-content: center;
  letter-spacing: 0;
  text-transform: none;
}

.settings-grid {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 0 12px 8px;
}

.slider-label-row {
  padding: 0 12px;
}

.control-section :deep(.v-slider) {
  padding: 0 12px 10px;
}

.control-section :deep(.v-slider__container) {
  min-height: 28px;
}

.upload-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.upload-toggle {
  padding: 0;
  min-height: unset;
  font-size: 0.85rem;
  text-transform: none;
}

.motion-progress-no-animation,
.motion-progress-no-animation *,
.motion-progress-no-animation::before,
.motion-progress-no-animation::after {
  transition: none !important;
  animation: none !important;
}

.motion-progress-no-animation :deep(.v-progress-linear__determinate),
.motion-progress-no-animation :deep(.v-progress-linear__indeterminate),
.motion-progress-no-animation :deep(.v-progress-linear__background) {
  transition: none !important;
  animation: none !important;
}

/* ── Mobile controls ── */
.mobile-joystick {
  position: fixed;
  bottom: 24px;
  left: 16px;
  z-index: 1100;
  touch-action: none;
}

.joystick-base {
  width: 90px;
  height: 90px;
  border-radius: 50%;
  background: rgba(30, 41, 59, 0.45);
  border: 2px solid rgba(255, 255, 255, 0.12);
  display: flex;
  align-items: center;
  justify-content: center;
}

.joystick-knob {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.25);
  border: 2px solid rgba(255, 255, 255, 0.35);
  transition: transform 0.05s ease-out;
}

.mobile-turn-buttons {
  position: fixed;
  bottom: 24px;
  right: 16px;
  display: flex;
  gap: 8px;
  z-index: 1100;
}

.mobile-turn-btn {
  width: 50px;
  height: 50px;
  border-radius: 50%;
  border: 2px solid rgba(255, 255, 255, 0.12);
  background: rgba(30, 41, 59, 0.45);
  color: #e2e8f0;
  font-size: 1.3rem;
  display: flex;
  align-items: center;
  justify-content: center;
  touch-action: manipulation;
}

.mobile-turn-btn:active {
  background: rgba(238, 99, 99, 0.5);
}

/* Skill switching bar (top-center) */
.mobile-skill-bar {
  position: fixed;
  top: 48px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 6px;
  z-index: 1100;
}

.mobile-skill-btn {
  padding: 6px 14px;
  border-radius: 16px;
  border: 1.5px solid rgba(255, 255, 255, 0.15);
  background: rgba(30, 41, 59, 0.5);
  color: #cbd5e1;
  font-size: 0.75rem;
  font-weight: 600;
  touch-action: manipulation;
  transition: background 0.15s ease, border-color 0.15s ease;
}

.mobile-skill-btn.active {
  background: rgba(238, 99, 99, 0.7);
  border-color: rgba(255, 176, 176, 0.6);
  color: #ffffff;
}

.mobile-skill-btn:active {
  background: rgba(238, 99, 99, 0.5);
}

/* Drawer toggle button (top-right) */
.mobile-drawer-toggle {
  position: fixed;
  top: 8px;
  right: 8px;
  z-index: 1200;
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 6px 14px;
  border: none;
  border-radius: 20px;
  background: rgba(30, 41, 59, 0.8);
  color: #e2e8f0;
  font-size: 0.78rem;
  font-weight: 600;
  backdrop-filter: blur(6px);
  touch-action: manipulation;
}

.mobile-drawer-chevron {
  font-size: 0.6rem;
  transition: transform 0.2s ease;
  transform: rotate(180deg);
}

.mobile-drawer-chevron.open {
  transform: rotate(0deg);
}

/* Slide-up drawer */
.mobile-drawer {
  position: fixed;
  top: 40px;
  right: 8px;
  left: 8px;
  z-index: 1150;
  max-height: calc(100vh - 180px);
  overflow-y: auto;
  overscroll-behavior: contain;
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.92);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.mobile-drawer-content {
  padding: 12px;
}

.mobile-drawer-section {
  margin-bottom: 10px;
}

.mobile-drawer-section:last-child {
  margin-bottom: 0;
}

.mobile-drawer-label {
  font-size: 0.7rem;
  font-weight: 700;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 6px;
}

.mobile-drawer-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.mobile-task-btn {
  padding: 6px 14px;
  border: none;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.1);
  color: #e2e8f0;
  font-size: 0.75rem;
  font-weight: 600;
  touch-action: manipulation;
}

.mobile-task-btn:disabled {
  opacity: 0.4;
}

.mobile-task-btn.active {
  background: #EE6363;
  color: #fff;
}

.mobile-task-toggle {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 6px;
  margin-bottom: 8px;
}

.mobile-reset-btn {
  background: rgba(185, 28, 28, 0.6);
}

.mobile-obj-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 6px;
  padding: 4px 0;
}

.mobile-obj-name {
  font-size: 0.75rem;
  color: #cbd5e1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.mobile-confirm-btn {
  background: rgba(34, 197, 94, 0.3);
  flex-shrink: 0;
  margin-left: 8px;
}

.mobile-obj-item-card {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  padding: 6px 10px;
  margin-top: 6px;
}

.mobile-obj-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  cursor: pointer;
}

.mobile-obj-controls {
  margin-top: 6px;
}

.mobile-obj-grid {
  display: grid;
  grid-template-columns: 28px 1fr 1fr 1fr;
  gap: 3px 4px;
  align-items: center;
  margin-bottom: 4px;
}

.mobile-obj-grid-label {
  font-size: 0.65rem;
  font-weight: 700;
  color: #94a3b8;
}

.mobile-obj-input {
  width: 100%;
  padding: 3px 4px;
  font-size: 0.7rem;
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 4px;
  text-align: center;
  background: rgba(255, 255, 255, 0.06);
  color: #e2e8f0;
  -moz-appearance: textfield;
}

.mobile-obj-input::-webkit-inner-spin-button,
.mobile-obj-input::-webkit-outer-spin-button {
  -webkit-appearance: none;
  margin: 0;
}

.mobile-obj-input:disabled {
  opacity: 0.4;
}

.mobile-obj-slider-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 2px;
}

.mobile-obj-slider-label {
  font-size: 0.65rem;
  color: #94a3b8;
  white-space: nowrap;
  min-width: 70px;
}

.mobile-obj-range {
  flex: 1;
  height: 4px;
  -webkit-appearance: none;
  appearance: none;
  background: rgba(255, 255, 255, 0.15);
  border-radius: 2px;
  outline: none;
}

.mobile-obj-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #EE6363;
  border: 2px solid rgba(255, 255, 255, 0.3);
}

.mobile-obj-range:disabled {
  opacity: 0.4;
}

/* Drawer transition */
.drawer-slide-enter-active,
.drawer-slide-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.drawer-slide-enter-from,
.drawer-slide-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

</style>
