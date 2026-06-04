import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const propOrange = new THREE.Color(0xf2731f);
const visualPalette = {
  red: new THREE.Color(0xf26a5b),
  green: new THREE.Color(0x33b982),
  blue: new THREE.Color(0x3f8fd8),
  yellow: new THREE.Color(0xe4bf3f),
  neutral: new THREE.Color(0x7c8ea3)
};

const cards = Array.from(document.querySelectorAll('[data-glb-src]'));
if (cards.length > 0) {
  const loader = new GLTFLoader();
  const previews = new Map();
  const clock = new THREE.Clock();
  let animationFrameId = null;

  cards.forEach((card) => {
    const stage = card.querySelector('.dataset-glb-stage');
    const src = card.getAttribute('data-glb-src');
    if (!stage || !src) return;
    previews.set(stage, createPreview(stage, src, loader));
  });

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        const preview = previews.get(entry.target);
        if (!preview) return;
        preview.setVisible(entry.isIntersecting);
        if (entry.isIntersecting) {
          preview.load();
          requestAnimation();
        }
      });
    }, { rootMargin: '220px 0px', threshold: 0.01 });

    previews.forEach((preview, stage) => observer.observe(stage));
  } else {
    previews.forEach((preview) => {
      preview.setVisible(true);
      preview.load();
    });
    requestAnimation();
  }

  function requestAnimation() {
    if (animationFrameId !== null) return;
    clock.getDelta();
    animationFrameId = requestAnimationFrame(animate);
  }

  function animate() {
    animationFrameId = null;
    const delta = clock.getDelta();
    let hasActivePreview = false;

    previews.forEach((preview) => {
      if (!preview.visible || !preview.renderer) return;
      hasActivePreview = true;
      preview.mixer?.update(delta);
      preview.renderer.render(preview.scene, preview.camera);
    });

    if (hasActivePreview) {
      animationFrameId = requestAnimationFrame(animate);
    }
  }
}

function createPreview(stage, src, loader) {
  let scene = null;
  let camera = null;
  let renderer = null;
  let mixer = null;
  let resizeObserver = null;
  let ready = null;
  let visible = false;

  const preview = {
    get scene() { return scene; },
    get camera() { return camera; },
    get renderer() { return renderer; },
    get mixer() { return mixer; },
    get visible() { return visible; },
    setVisible(value) {
      visible = value;
    },
    load() {
      if (ready) return ready;
      ready = initializePreview();
      return ready;
    }
  };

  function initializePreview() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf6f7f9);

    camera = new THREE.PerspectiveCamera(34, 1, 0.03, 80);
    camera.position.set(2.6, 1.6, 3.0);

    renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
      precision: 'highp'
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 3));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    stage.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x8b929e, 0.9));
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.34);
    keyLight.position.set(5, 7, 4);
    scene.add(keyLight);

    const groundTexture = makeCheckerTexture();
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(8, 8),
      new THREE.MeshStandardMaterial({ map: groundTexture, roughness: 0.86 })
    );
    ground.rotation.x = -Math.PI / 2;
    scene.add(ground);

    const resize = () => {
      const rect = stage.getBoundingClientRect();
      const width = Math.max(1, Math.floor(rect.width));
      const height = Math.max(1, Math.floor(rect.height));
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    resize();

    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(stage);

    return loadGltf(loader, src)
      .then((gltf) => {
        const model = gltf.scene;
        polishVisualizationColors(model);
        scene.add(model);

        if (gltf.animations.length > 0) {
          mixer = new THREE.AnimationMixer(model);
          gltf.animations.forEach((clip) => {
            const action = mixer.clipAction(clip);
            action.play();
          });
        }
        const bounds = measurePreviewBounds(model, mixer, gltf.animations);
        fitCameraToBounds(bounds, camera);
        if (mixer) mixer.setTime(0);
        resize();
      })
      .catch((error) => {
        stage.classList.add('is-error');
        resizeObserver?.disconnect();
        console.warn('Failed to load dataset GLB preview: ' + src, error);
      });
  }

  return preview;
}

function loadGltf(loader, url) {
  return new Promise((resolve, reject) => {
    loader.load(url, resolve, undefined, reject);
  });
}

function measurePreviewBounds(model, mixer, animations) {
  const bounds = new THREE.Box3();
  expandBounds(bounds, model);

  if (mixer && animations.length > 0) {
    const duration = Math.max(...animations.map((clip) => clip.duration || 0));
    const sampleCount = 16;
    for (let i = 0; i <= sampleCount; i += 1) {
      mixer.setTime((duration * i) / sampleCount);
      expandBounds(bounds, model);
    }
  }

  return bounds;
}

function expandBounds(bounds, model) {
  model.updateMatrixWorld(true);
  const frameBounds = new THREE.Box3().setFromObject(model);
  if (!frameBounds.isEmpty()) bounds.union(frameBounds);
}

function fitCameraToBounds(box, camera) {
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);

  const radius = Math.max(box.getBoundingSphere(new THREE.Sphere()).radius, 1.0);
  const verticalFov = THREE.MathUtils.degToRad(camera.fov);
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * camera.aspect);
  const distance = Math.max(
    radius / Math.sin(verticalFov / 2),
    radius / Math.sin(horizontalFov / 2)
  ) * 0.944;
  const direction = new THREE.Vector3(1.08, 0.54, 1.2).normalize();
  const target = center.clone();
  target.y += Math.max(size.y * 0.04, 0.05);

  camera.position.copy(target).addScaledVector(direction, distance);
  camera.lookAt(target);
  camera.near = Math.max(0.01, distance / 160);
  camera.far = Math.max(30, distance * 8);
  camera.updateProjectionMatrix();
}

function polishVisualizationColors(model) {
  model.traverse((child) => {
    if (!child.isMesh || (!isPropObject(child) && !isVisualOverlayObject(child))) return;
    child.castShadow = false;
    child.receiveShadow = false;
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    const polishedMaterials = materials.map((material) => {
      if (!material) return material;
      const clone = material.clone();
      const color = isPropObject(child) ? propOrange : pickPolishedColor(clone.color);
      if (clone.color) clone.color.copy(color);
      if (clone.emissive && clone.color) {
        if (isPropObject(child)) {
          clone.emissive.copy(color).multiplyScalar(0.22);
          clone.emissiveIntensity = 0.6;
        } else {
          clone.emissive.copy(color).multiplyScalar(isVisualOverlayObject(child) ? 0.38 : 0.22);
          clone.emissiveIntensity = isVisualOverlayObject(child) ? 0.9 : 0.65;
        }
      }
      if (isVisualOverlayObject(child)) {
        clone.transparent = true;
        clone.opacity = 0.96;
      }
      if (isPropObject(child)) {
        clone.transparent = false;
        clone.opacity = 1.0;
        clone.alphaTest = 0;
        clone.alphaMap = null;
        clone.side = THREE.DoubleSide;
        clone.depthWrite = true;
        clone.depthTest = true;
        if ('roughness' in clone) clone.roughness = 0.9;
        if ('metalness' in clone) clone.metalness = 0.0;
      }
      if ('metalness' in clone) clone.metalness = 0.0;
      clone.needsUpdate = true;
      return clone;
    });
    child.material = Array.isArray(child.material) ? polishedMaterials : polishedMaterials[0];
  });
}

function pickPolishedColor(color) {
  if (!color) return visualPalette.neutral.clone();
  const { r, g, b } = color;
  if (r > 0.55 && g > 0.45 && b < 0.35) return visualPalette.yellow.clone();
  if (g >= r && g >= b) return visualPalette.green.clone();
  if (b >= r && b >= g) return visualPalette.blue.clone();
  if (r >= g && r >= b) return visualPalette.red.clone();
  return visualPalette.neutral.clone();
}

function isBoxObject(object) {
  const names = collectNames(object);
  return /(^|[_\s])box($|[_\s\d])/.test(names) || names.includes('carrybox') || names.includes('object_box');
}

function isBallObject(object) {
  const names = collectNames(object);
  return names.includes('soccer') || names.includes('ball');
}

function isPropObject(object) {
  return isBoxObject(object) || isBallObject(object);
}

function isVisualOverlayObject(object) {
  const names = collectNames(object);
  return names.includes('visual_') || names.includes('seg_') || names.includes('trajectory') || names.includes('path');
}

function collectNames(object) {
  return [
    object.name || '',
    object.parent?.name || '',
    object.material?.name || '',
    ...(Array.isArray(object.material) ? object.material.map((material) => material?.name || '') : [])
  ].join(' ').toLowerCase();
}

function makeCheckerTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#f0f2f5';
  ctx.fillRect(0, 0, 128, 128);
  ctx.fillStyle = '#e3e7ed';
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      if ((x + y) % 2 === 0) ctx.fillRect(x * 16, y * 16, 16, 16);
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(6, 6);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
