import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

// ============================================================ constantes
const PALETTE = {
  ego: 0xf97316,
  success: 0x22c55e,
  collision: 0xef4444,
  sensor: 0x38bdf8,
  parked: [0x9ca3af, 0x6b7280, 0x4b5563, 0xd1d5db, 0x8e9aaf, 0xa39e93],
};
const AGENT_LABELS = { q_learning: "Q-learning", neural: "Red neuronal", random: "Aleatorio" };
const STATUS_LABELS = {
  idle: "En espera", running: "En curso", paused: "Pausado",
  success: "Estacionado", collision: "Colisión", timeout: "Tiempo agotado",
};
const CAMERA_LABELS = { orbit: "orbital", top: "superior", follow: "seguimiento" };
const BASE_STEP_MS = 200;      // dt del ambiente a velocidad 1×
const AUTO_RESET_MS = 1800;
const WHEEL_RADIUS = 0.34;
const BRAKE_ACTION = 4;

const $ = (id) => document.getElementById(id);

// ============================================================ conexión
class Connection {
  constructor(onStatus) {
    this.onStatus = onStatus;
    this.pending = [];
    this.connect();
  }

  connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/ws`);
    this.ready = new Promise((resolve) => (this.ws.onopen = () => { this.onStatus(true); resolve(); }));
    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      const pending = this.pending.shift();
      if (!pending) return;
      message.type === "error" ? pending.reject(new Error(message.message)) : pending.resolve(message);
    };
    this.ws.onclose = () => {
      this.onStatus(false);
      this.pending.splice(0).forEach((p) => p.reject(new Error("Conexión perdida")));
      setTimeout(() => this.connect(), 2000);
    };
  }

  // El servidor responde en orden, así que cada respuesta corresponde a la petición más antigua.
  async send(cmd, payload = {}) {
    await this.ready;
    return new Promise((resolve, reject) => {
      this.pending.push({ resolve, reject });
      this.ws.send(JSON.stringify({ cmd, payload }));
    });
  }
}

// ============================================================ escena
const viewport = $("viewport");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
viewport.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0d14);
scene.fog = new THREE.Fog(0x0a0d14, 45, 110);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 300);
camera.position.set(0, 24, 26);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI / 2.1;
controls.minDistance = 6;
controls.maxDistance = 70;

scene.add(new THREE.HemisphereLight(0xa8c4ff, 0x1a1d24, 0.7));
const sun = new THREE.DirectionalLight(0xfff1dd, 1.6);
sun.position.set(-14, 26, 12);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -24, right: 24, top: 18, bottom: -18, near: 1, far: 70 });
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.03;
scene.add(sun);

// env (x, y) del simulador → (x, 0, -y) en Three.js; el rumbo θ es rotation.y
const toWorld = (x, y, h = 0) => new THREE.Vector3(x, h, -y);

function std(params) {
  return new THREE.MeshStandardMaterial({ envMapIntensity: 0.3, ...params });
}

function shadowed(mesh, cast = true, receive = true) {
  mesh.castShadow = cast;
  mesh.receiveShadow = receive;
  return mesh;
}

function canvasTexture(width, height, draw) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  draw(canvas.getContext("2d"), width, height);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

function asphaltTexture() {
  const texture = canvasTexture(1024, 1024, (ctx, w, h) => {
    ctx.fillStyle = "#34373d";
    ctx.fillRect(0, 0, w, h);
    const image = ctx.getImageData(0, 0, w, h);
    for (let i = 0; i < image.data.length; i += 4) {
      const n = (Math.random() - 0.5) * 28;
      image.data[i] += n; image.data[i + 1] += n; image.data[i + 2] += n;
    }
    ctx.putImageData(image, 0, 0);
    for (let i = 0; i < 26; i++) {            // manchas de aceite
      const x = Math.random() * w, y = Math.random() * h, r = 20 + Math.random() * 70;
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, "rgba(10,10,12,0.32)");
      g.addColorStop(1, "rgba(10,10,12,0)");
      ctx.fillStyle = g;
      ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
    }
    ctx.strokeStyle = "rgba(15,15,18,0.5)";   // grietas
    for (let i = 0; i < 14; i++) {
      ctx.lineWidth = 1 + Math.random() * 1.5;
      ctx.beginPath();
      let x = Math.random() * w, y = Math.random() * h;
      ctx.moveTo(x, y);
      for (let j = 0; j < 8; j++) ctx.lineTo((x += (Math.random() - 0.5) * 60), (y += (Math.random() - 0.5) * 60));
      ctx.stroke();
    }
  });
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(3, 2);
  return texture;
}

function hazardTexture() {
  const texture = canvasTexture(128, 128, (ctx, w, h) => {
    ctx.fillStyle = "#facc15";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#111";
    for (let i = -h; i < w; i += 40) {
      ctx.beginPath();
      ctx.moveTo(i, h); ctx.lineTo(i + 20, h); ctx.lineTo(i + 20 + h, 0); ctx.lineTo(i + h, 0);
      ctx.fill();
    }
  });
  texture.wrapS = THREE.RepeatWrapping;
  texture.repeat.set(2, 1);
  return texture;
}

function labelTexture(text, { color = "#e5e7eb", bg = null, size = 128, font = 72 } = {}) {
  return canvasTexture(size * 2, size, (ctx, w, h) => {
    if (bg) {
      ctx.fillStyle = bg;
      ctx.beginPath();
      ctx.roundRect(4, 4, w - 8, h - 8, 24);
      ctx.fill();
    }
    ctx.fillStyle = color;
    ctx.font = `700 ${font}px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, w / 2, h / 2 + 4);
  });
}

function flatPlane(width, depth, material, x, y, height = 0.01) {
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, depth), material);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.copy(toWorld(x, y, height));
  mesh.receiveShadow = true;
  return mesh;
}

// ============================================================ parqueadero estático
const paint = std({ color: 0xf1f1ee, roughness: 0.6, emissive: 0x2a2a2a });
const yellowPaint = std({ color: 0xfacc15, roughness: 0.6, emissive: 0x2a2205 });
const concrete = std({ color: 0x9aa0a8, roughness: 0.9 });
let lotBuilt = false;

function buildLot(layout) {
  const { width: W, height: H } = layout.lot;
  const group = new THREE.Group();

  const ground = flatPlane(220, 220, std({ color: 0x0f141c, roughness: 1 }), 0, 0, -0.02);
  group.add(ground);
  group.add(flatPlane(W + 7, H + 7, std({ color: 0x5c6169, roughness: 0.95 }), 0, 0, -0.01));
  group.add(flatPlane(W, H, std({ map: asphaltTexture(), roughness: 0.92 }), 0, 0, 0));

  // Bordillos con segmentos alternados amarillo/gris
  const curbMats = [std({ color: 0xd4d4d4, roughness: 0.8 }), std({ color: 0xeab308, roughness: 0.7 })];
  const addCurbLine = (x0, y0, x1, y1) => {
    const length = Math.hypot(x1 - x0, y1 - y0), n = Math.round(length / 1.5);
    const angle = Math.atan2(y1 - y0, x1 - x0);
    for (let i = 0; i < n; i++) {
      const t = (i + 0.5) / n;
      const curb = shadowed(new THREE.Mesh(new THREE.BoxGeometry(length / n - 0.04, 0.2, 0.32), curbMats[i % 2]));
      curb.position.copy(toWorld(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, 0.1));
      curb.rotation.y = angle;
      group.add(curb);
    }
  };
  const cw = W / 2 + 0.16, ch = H / 2 + 0.16;
  addCurbLine(-cw, ch, cw, ch);
  addCurbLine(-cw, -ch, cw, -ch);
  addCurbLine(cw, -ch, cw, ch);
  addCurbLine(-cw, -ch, -cw, ch);

  // Líneas de los cupos, topes de rueda y numeración
  const rows = [0, 1].map((r) => layout.slots.filter((s) => s.row === r));
  rows.forEach((slots, r) => {
    const { width: sw, depth: sd } = slots[0];
    const sign = r === 0 ? 1 : -1;
    const inner = sign * (H / 2 - sd), outer = sign * (H / 2);
    const xs = slots.map((s) => s.x);
    for (let i = 0; i <= slots.length; i++) {
      const x = Math.min(...xs) - sw / 2 + i * sw;
      group.add(flatPlane(0.12, sd, paint, x, (inner + outer) / 2, 0.012));
      group.add(flatPlane(0.5, 0.12, paint, x, inner, 0.012));   // remate en "T"
    }
    group.add(flatPlane(slots.length * sw, 0.12, paint, 0, outer - sign * 0.06, 0.012));
    slots.forEach((slot, i) => {
      const stop = shadowed(new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.12, 0.18), concrete));
      stop.position.copy(toWorld(slot.x, outer - sign * 0.55, 0.06));
      group.add(stop);
      const label = new THREE.Mesh(
        new THREE.PlaneGeometry(1.4, 0.7),
        new THREE.MeshBasicMaterial({ map: labelTexture(`${"AB"[r]}${i + 1}`, { color: "#d6d6d6" }),
          transparent: true, opacity: 0.85, depthWrite: false }),
      );
      label.rotation.set(-Math.PI / 2, 0, r === 0 ? 0 : Math.PI);
      label.position.copy(toWorld(slot.x, inner + sign * 0.8, 0.014));
      group.add(label);
    });
  });

  // Pasillo: línea central discontinua y flechas de sentido
  for (let x = -W / 2 + 2; x < W / 2 - 2; x += 2) group.add(flatPlane(1.1, 0.14, yellowPaint, x + 0.55, 0, 0.012));
  const arrowShape = new THREE.Shape();
  arrowShape.moveTo(-1.1, -0.18); arrowShape.lineTo(0.2, -0.18); arrowShape.lineTo(0.2, -0.5);
  arrowShape.lineTo(1.1, 0); arrowShape.lineTo(0.2, 0.5); arrowShape.lineTo(0.2, 0.18);
  arrowShape.lineTo(-1.1, 0.18); arrowShape.closePath();
  const arrowGeo = new THREE.ShapeGeometry(arrowShape);
  for (const x of [-9, 0, 9]) {
    for (const [lane, dir] of [[1.9, Math.PI], [-1.9, 0]]) {
      const arrow = new THREE.Mesh(arrowGeo, paint);
      arrow.rotation.set(-Math.PI / 2, 0, dir);
      arrow.position.copy(toWorld(x, lane, 0.013));
      arrow.receiveShadow = true;
      group.add(arrow);
    }
  }

  // Postes de luz
  const poleMat = std({ color: 0x2b3240, metalness: 0.6, roughness: 0.4 });
  const lampMat = new THREE.MeshBasicMaterial({ color: 0xfff3d1 });
  for (const [x, y] of [[-W / 2 - 1.6, H / 2 + 1.6], [W / 2 + 1.6, H / 2 + 1.6], [-W / 2 - 1.6, -H / 2 - 1.6],
    [W / 2 + 1.6, -H / 2 - 1.6], [-W / 2 - 1.8, 0], [W / 2 + 1.8, 0]]) {
    const pole = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.13, 6, 12), poleMat));
    pole.position.copy(toWorld(x, y, 3));
    const dirToCenter = Math.atan2(-y, -x);
    const arm = shadowed(new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.08, 0.08), poleMat));
    arm.position.copy(toWorld(x + Math.cos(dirToCenter) * 0.7, y + Math.sin(dirToCenter) * 0.7, 6));
    arm.rotation.y = dirToCenter;
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.12, 0.3), lampMat);
    head.position.copy(toWorld(x + Math.cos(dirToCenter) * 1.35, y + Math.sin(dirToCenter) * 1.35, 5.92));
    head.rotation.y = dirToCenter;
    const light = new THREE.PointLight(0xffe4b5, 55, 26, 1.7);
    light.position.copy(head.position).add(new THREE.Vector3(0, -0.3, 0));
    group.add(pole, arm, head, light);
  }

  // Árboles alrededor
  const trunkMat = std({ color: 0x4a3426, roughness: 0.9 });
  const leafMats = [std({ color: 0x1f4d34, roughness: 0.8 }), std({ color: 0x2a5c3a, roughness: 0.8 })];
  for (let i = 0; i < 16; i++) {
    const side = i % 4;
    const t = (Math.floor(i / 4) + 0.5) / 4;
    const x = side < 2 ? -W / 2 + W * t + 1.5 : (side === 2 ? -1 : 1) * (W / 2 + 5);
    const y = side < 2 ? (side === 0 ? 1 : -1) * (H / 2 + 5) : -H / 2 + H * t;
    const s = 0.8 + ((i * 37) % 10) / 20;
    const trunk = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(0.15 * s, 0.22 * s, 1.6 * s, 8), trunkMat));
    trunk.position.copy(toWorld(x, y, 0.8 * s));
    const leaves = shadowed(new THREE.Mesh(new THREE.IcosahedronGeometry(1.3 * s, 1), leafMats[i % 2]));
    leaves.position.copy(toWorld(x, y, 2.3 * s));
    group.add(trunk, leaves);
  }

  scene.add(group);
  lotBuilt = true;
}

// ============================================================ vehículos
const carGeometry = (() => {
  const L = 2.03;  // medio largo antes del bisel
  const body = new THREE.Shape();
  body.moveTo(-L, 0.3);
  body.lineTo(-1.72, 0.3);
  body.absarc(-1.3, 0.3, 0.42, Math.PI, 0, true);
  body.lineTo(0.88, 0.3);
  body.absarc(1.3, 0.3, 0.42, Math.PI, 0, true);
  body.lineTo(L, 0.3);
  body.lineTo(L + 0.02, 0.6);
  body.quadraticCurveTo(L, 0.8, 1.8, 0.84);
  body.lineTo(0.72, 0.93);
  body.lineTo(-1.55, 0.96);
  body.lineTo(-1.95, 0.9);
  body.quadraticCurveTo(-L - 0.05, 0.84, -L, 0.6);
  body.closePath();
  const bodyGeo = new THREE.ExtrudeGeometry(body, {
    depth: 1.66, bevelEnabled: true, bevelThickness: 0.07, bevelSize: 0.07, bevelSegments: 4, curveSegments: 18,
  });
  bodyGeo.translate(0, 0, -0.83);

  const cabin = new THREE.Shape();
  cabin.moveTo(0.72, 0.92);
  cabin.lineTo(0.0, 1.36);
  cabin.lineTo(-0.98, 1.38);
  cabin.lineTo(-1.62, 0.95);
  cabin.closePath();
  const cabinGeo = new THREE.ExtrudeGeometry(cabin, {
    depth: 1.36, bevelEnabled: true, bevelThickness: 0.05, bevelSize: 0.05, bevelSegments: 3,
  });
  cabinGeo.translate(0, 0, -0.68);

  const tire = new THREE.CylinderGeometry(WHEEL_RADIUS, WHEEL_RADIUS, 0.24, 28).rotateX(Math.PI / 2);
  const rim = new THREE.CylinderGeometry(0.22, 0.22, 0.25, 20).rotateX(Math.PI / 2);
  const spoke = new THREE.BoxGeometry(0.4, 0.05, 0.26);
  const arrowShape = new THREE.Shape();
  arrowShape.moveTo(0.7, 0); arrowShape.lineTo(-0.15, 0.45); arrowShape.lineTo(0.05, 0);
  arrowShape.lineTo(-0.15, -0.45); arrowShape.closePath();
  return { body: bodyGeo, cabin: cabinGeo, tire, rim, spoke, arrow: new THREE.ShapeGeometry(arrowShape),
    box: new THREE.BoxGeometry(1, 1, 1) };
})();

const sharedMats = {
  glass: new THREE.MeshPhysicalMaterial({ color: 0x0c1622, metalness: 0.2, roughness: 0.08, clearcoat: 1, envMapIntensity: 0.9 }),
  tire: std({ color: 0x15171b, roughness: 0.85 }),
  rim: std({ color: 0xc7ccd4, metalness: 0.9, roughness: 0.25, envMapIntensity: 0.8 }),
  trim: std({ color: 0x1b1f26, roughness: 0.6 }),
  plate: std({ color: 0xf5f5f0, roughness: 0.5 }),
};
const paintCache = new Map();
function paintMaterial(color) {
  if (!paintCache.has(color)) {
    paintCache.set(color, new THREE.MeshPhysicalMaterial({
      color, metalness: 0.45, roughness: 0.32, clearcoat: 1, clearcoatRoughness: 0.08, envMapIntensity: 0.7,
    }));
  }
  return paintCache.get(color);
}

function boxPart(material, size, position) {
  const mesh = shadowed(new THREE.Mesh(carGeometry.box, material));
  mesh.scale.set(...size);
  mesh.position.set(...position);
  return mesh;
}

function makeCar({ color, ego = false }) {
  const group = new THREE.Group();
  const bodyMat = ego ? paintMaterial(color).clone() : paintMaterial(color);
  group.add(shadowed(new THREE.Mesh(carGeometry.body, bodyMat)));
  group.add(shadowed(new THREE.Mesh(carGeometry.cabin, sharedMats.glass)));
  group.add(boxPart(bodyMat, [0.98, 0.05, 1.44], [-0.48, 1.41, 0]));         // techo
  group.add(boxPart(bodyMat, [0.08, 0.44, 1.48], [-0.42, 1.16, 0]));         // pilar B
  group.add(boxPart(sharedMats.trim, [0.18, 0.2, 1.74], [2.02, 0.36, 0]));   // parachoques
  group.add(boxPart(sharedMats.trim, [0.18, 0.2, 1.74], [-2.02, 0.36, 0]));
  group.add(boxPart(sharedMats.trim, [0.05, 0.16, 0.72], [2.09, 0.56, 0]));  // parrilla
  group.add(boxPart(sharedMats.plate, [0.02, 0.12, 0.42], [2.12, 0.38, 0]));
  group.add(boxPart(sharedMats.plate, [0.02, 0.12, 0.42], [-2.12, 0.38, 0]));
  for (const z of [-0.95, 0.95]) group.add(boxPart(bodyMat, [0.18, 0.1, 0.14], [0.62, 0.98, z]));  // espejos

  const headMat = new THREE.MeshStandardMaterial({ color: 0xfff6e0, emissive: 0xfff1d0, emissiveIntensity: ego ? 2.5 : 0.15 });
  const tailMat = new THREE.MeshStandardMaterial({ color: 0x7f1010, emissive: 0xff1a1a, emissiveIntensity: ego ? 0.8 : 0.1 });
  const reverseMat = new THREE.MeshStandardMaterial({ color: 0xdddddd, emissive: 0xffffff, emissiveIntensity: 0 });
  for (const z of [-0.6, 0.6]) {
    group.add(boxPart(headMat, [0.06, 0.12, 0.4], [2.07, 0.68, z]));
    group.add(boxPart(tailMat, [0.06, 0.13, 0.42], [-2.09, 0.74, z * 0.95]));
    group.add(boxPart(reverseMat, [0.06, 0.08, 0.14], [-2.09, 0.56, z * 0.55]));
  }

  // Ruedas: el pivote gira (dirección) y la rueda rota (avance)
  const wheels = [], steerPivots = [];
  for (const [x, z] of [[1.3, 0.78], [1.3, -0.78], [-1.3, 0.78], [-1.3, -0.78]]) {
    const pivot = new THREE.Group();
    pivot.position.set(x, WHEEL_RADIUS, z);
    const wheel = new THREE.Group();
    wheel.add(shadowed(new THREE.Mesh(carGeometry.tire, sharedMats.tire)));
    wheel.add(shadowed(new THREE.Mesh(carGeometry.rim, sharedMats.rim), false));
    for (let i = 0; i < 5; i++) {
      const spoke = new THREE.Mesh(carGeometry.spoke, sharedMats.trim);
      spoke.rotation.z = (i / 5) * Math.PI;
      wheel.add(spoke);
    }
    pivot.add(wheel);
    group.add(pivot);
    wheels.push(wheel);
    if (x > 0) steerPivots.push(pivot);
  }
  return { group, bodyMat, wheels, steerPivots, tailMat, reverseMat };
}

// ============================================================ vehículo controlado
const ego = makeCar({ color: PALETTE.ego, ego: true });
scene.add(ego.group);

const egoArrow = new THREE.Mesh(carGeometry.arrow,
  new THREE.MeshBasicMaterial({ color: 0xffd7b5, transparent: true, opacity: 0.95, side: THREE.DoubleSide }));
egoArrow.rotation.x = -Math.PI / 2;
const arrowHolder = new THREE.Group();
arrowHolder.position.y = 2.0;
arrowHolder.add(egoArrow);
ego.group.add(arrowHolder);

const egoGlow = new THREE.Mesh(new THREE.CircleGeometry(2.6, 48),
  new THREE.MeshBasicMaterial({ color: PALETTE.ego, transparent: true, opacity: 0.12, depthWrite: false }));
egoGlow.rotation.x = -Math.PI / 2;
egoGlow.position.y = 0.02;
egoGlow.scale.set(1.15, 0.6, 1);
ego.group.add(egoGlow);

for (const z of [-0.6, 0.6]) {
  const spot = new THREE.SpotLight(0xfff1d0, 30, 16, 0.5, 0.6, 1.4);
  spot.position.set(2.0, 0.7, z);
  spot.target.position.set(8, 0, z * 2);
  ego.group.add(spot, spot.target);
}

const collisionRing = new THREE.Mesh(new THREE.RingGeometry(2.5, 2.8, 64),
  new THREE.MeshBasicMaterial({ color: PALETTE.collision, transparent: true, opacity: 0, depthWrite: false }));
collisionRing.rotation.x = -Math.PI / 2;
collisionRing.position.y = 0.03;
ego.group.add(collisionRing);

// Sensores: un haz y un punto de impacto por rayo, en el marco local del vehículo
const sensorGroup = new THREE.Group();
sensorGroup.position.y = 0.55;
ego.group.add(sensorGroup);
const beamGeo = new THREE.BoxGeometry(1, 0.035, 0.035).translate(0.5, 0, 0);
const hitGeo = new THREE.SphereGeometry(0.11, 12, 8);
const sensors = [];
function ensureSensors(count) {
  while (sensors.length < count) {
    const material = new THREE.MeshBasicMaterial({ color: PALETTE.sensor, transparent: true, opacity: 0.75, depthWrite: false });
    const beam = new THREE.Mesh(beamGeo, material);
    const hit = new THREE.Mesh(hitGeo, material);
    sensorGroup.add(beam, hit);
    sensors.push({ beam, hit, material });
  }
}

// ============================================================ cupo objetivo
const target = new THREE.Group();
const targetFillMat = new THREE.MeshBasicMaterial({ color: PALETTE.success, transparent: true, opacity: 0.25, depthWrite: false });
const targetFill = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), targetFillMat);
targetFill.rotation.x = -Math.PI / 2;
targetFill.position.y = 0.015;
target.add(targetFill);
const outlineMat = new THREE.MeshBasicMaterial({ color: 0x4ade80 });
const outline = [0, 1, 2, 3].map(() => {
  const strip = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), outlineMat);
  strip.rotation.x = -Math.PI / 2;
  strip.position.y = 0.018;
  target.add(strip);
  return strip;
});
const beamTexture = canvasTexture(4, 128, (ctx, w, h) => {
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, "rgba(34,197,94,0)");
  g.addColorStop(1, "rgba(34,197,94,0.55)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
});
const targetBeam = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 2.6, 4, 1, true).rotateY(Math.PI / 4),
  new THREE.MeshBasicMaterial({ map: beamTexture, transparent: true, depthWrite: false, side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending }));
targetBeam.position.y = 1.3;
target.add(targetBeam);
const pin = new THREE.Sprite(new THREE.SpriteMaterial({
  map: labelTexture("P", { color: "#ffffff", bg: "#16a34a", size: 96, font: 70 }), depthWrite: false,
}));
pin.scale.set(1.2, 0.6, 1);
target.add(pin);
scene.add(target);

function placeTarget(info, slot) {
  const w = slot.width - 0.25, d = slot.depth - 0.25, t = 0.12;
  target.position.copy(toWorld(info.x, info.y));
  targetFill.scale.set(w, d, 1);
  outline[0].scale.set(w, t, 1); outline[0].position.set(0, 0.018, -d / 2);
  outline[1].scale.set(w, t, 1); outline[1].position.set(0, 0.018, d / 2);
  outline[2].scale.set(t, d, 1); outline[2].position.set(-w / 2, 0.018, 0);
  outline[3].scale.set(t, d, 1); outline[3].position.set(w / 2, 0.018, 0);
  targetBeam.scale.set(w / Math.SQRT2, 1, d / Math.SQRT2);
}

// ============================================================ obstáculos del episodio
const obstacleGroup = new THREE.Group();
scene.add(obstacleGroup);
const coneMat = std({ color: 0xff6a13, roughness: 0.5, emissive: 0x3a1200 });
const coneBandMat = std({ color: 0xf5f5f5, roughness: 0.3, emissive: 0x222222 });
const pillarMat = std({ color: 0xa4a9b0, roughness: 0.85 });
const hazardMat = std({ map: hazardTexture(), roughness: 0.7 });
const coneGeo = new THREE.ConeGeometry(0.2, 0.72, 20);
const coneBandGeo = new THREE.CylinderGeometry(0.115, 0.14, 0.12, 20);

function buildObstacles(layout) {
  obstacleGroup.clear();
  for (const o of layout.obstacles) {
    if (o.kind === "car") {
      const car = makeCar({ color: PALETTE.parked[o.color % PALETTE.parked.length] });
      car.group.position.copy(toWorld(o.x, o.y));
      car.group.rotation.y = o.angle + (o.color % 2 ? Math.PI / 2 : -Math.PI / 2);
      obstacleGroup.add(car.group);
    } else if (o.kind === "cone") {
      const cone = new THREE.Group();
      cone.add(boxPart(coneMat, [0.5, 0.05, 0.5], [0, 0.025, 0]));
      const body = shadowed(new THREE.Mesh(coneGeo, coneMat));
      body.position.y = 0.41;
      const band = new THREE.Mesh(coneBandGeo, coneBandMat);
      band.position.y = 0.46;
      cone.add(body, band);
      cone.position.copy(toWorld(o.x, o.y));
      obstacleGroup.add(cone);
    } else if (o.kind === "pillar") {
      const pillar = new THREE.Group();
      pillar.add(boxPart(pillarMat, [o.hx * 2, 3.2, o.hy * 2], [0, 1.6, 0]));
      pillar.add(boxPart(hazardMat, [o.hx * 2 + 0.02, 0.7, o.hy * 2 + 0.02], [0, 0.35, 0]));
      pillar.position.copy(toWorld(o.x, o.y));
      obstacleGroup.add(pillar);
    }
  }
}

// ============================================================ trayectoria
const TRAIL_MAX = 800;
const trailPoints = [];
const trailMesh = new THREE.Mesh(new THREE.BufferGeometry(),
  new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false }));
trailMesh.position.y = 0.022;
scene.add(trailMesh);

function rebuildTrail() {
  const n = trailPoints.length;
  if (n < 2) { trailMesh.geometry.setDrawRange(0, 0); return; }
  const positions = new Float32Array(n * 6), colors = new Float32Array(n * 6), indices = [];
  const color = new THREE.Color(), dim = new THREE.Color(0x3a1d08), bright = new THREE.Color(PALETTE.ego);
  for (let i = 0; i < n; i++) {
    const a = trailPoints[Math.max(0, i - 1)], b = trailPoints[Math.min(n - 1, i + 1)];
    let nx = -(b.y - a.y), ny = b.x - a.x;
    const len = Math.hypot(nx, ny) || 1;
    nx = (nx / len) * 0.09; ny = (ny / len) * 0.09;
    const p = trailPoints[i];
    positions.set([p.x + nx, 0, -(p.y + ny), p.x - nx, 0, -(p.y - ny)], i * 6);
    color.copy(dim).lerp(bright, i / (n - 1));
    colors.set([color.r, color.g, color.b, color.r, color.g, color.b], i * 6);
    if (i < n - 1) indices.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
  }
  const geo = trailMesh.geometry;
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.setIndex(indices);
  geo.setDrawRange(0, indices.length);
  geo.computeBoundingSphere();
}

function pushTrail(x, y) {
  const last = trailPoints[trailPoints.length - 1];
  if (last && Math.hypot(last.x - x, last.y - y) < 0.12) return;
  trailPoints.push({ x, y });
  if (trailPoints.length > TRAIL_MAX) trailPoints.shift();
  rebuildTrail();
}

// ============================================================ estado de la simulación
const sim = {
  running: false, inFlight: false, lastStepAt: 0, episodeOver: false, autoResetAt: 0,
  speed: 1, camera: "orbit", showSensors: true, showTrail: true,
  layout: null, frame: null, carHalf: [2.1, 0.9],
  from: null, to: null, animStart: 0, animDuration: BASE_STEP_MS,
  pose: { x: 0, y: 0, heading: 0, steer: 0, sensors: [] },
  statusAt: 0,
};
const stepInterval = () => BASE_STEP_MS / sim.speed;

function lerpAngle(a, b, t) {
  const d = Math.atan2(Math.sin(b - a), Math.cos(b - a));
  return a + d * t;
}

function poseFromFrame(frame) {
  return {
    x: frame.car.x, y: frame.car.y, heading: frame.car.heading, steer: frame.car.steer,
    sensors: frame.sensors.map((s) => s.distance),
  };
}

function applyReset(frame) {
  if (!lotBuilt) buildLot(frame.layout);
  sim.layout = frame.layout;
  sim.carHalf = [frame.car.length / 2, frame.car.width / 2];
  buildObstacles(frame.layout);
  placeTarget(frame.target, frame.layout.slots.find((s) => s.id === frame.target.id));
  ensureSensors(frame.sensors.length);
  sim.pose = poseFromFrame(frame);
  sim.from = sim.to = sim.pose;
  sim.episodeOver = false;
  trailPoints.length = 0;
  pushTrail(frame.car.x, frame.car.y);
  hideBanner();
  applyFrame(frame, false);
}

function applyFrame(frame, animate = true) {
  sim.frame = frame;
  if (animate) {
    sim.from = { ...sim.pose, sensors: [...sim.pose.sensors] };
    sim.to = poseFromFrame(frame);
    sim.animStart = performance.now();
    sim.animDuration = stepInterval();
  }
  ego.tailMat.emissiveIntensity = frame.action === BRAKE_ACTION ? 3 : 0.8;
  ego.reverseMat.emissiveIntensity = frame.car.speed < -0.05 ? 2.5 : 0;
  if (["success", "collision", "timeout"].includes(frame.status) && !sim.episodeOver) {
    sim.episodeOver = true;
    sim.autoResetAt = performance.now() + AUTO_RESET_MS;
    sim.statusAt = performance.now();
    showBanner(frame.status);
  }
  updatePanel(frame);
  updateTraining(frame.training);
}

function updatePose(now) {
  if (!sim.to) return;
  const t = Math.min(1, (now - sim.animStart) / sim.animDuration);
  const { from, to } = sim;
  sim.pose = {
    x: from.x + (to.x - from.x) * t,
    y: from.y + (to.y - from.y) * t,
    heading: lerpAngle(from.heading, to.heading, t),
    steer: from.steer + (to.steer - from.steer) * t,
    sensors: to.sensors.map((d, i) => (from.sensors[i] ?? d) + (d - (from.sensors[i] ?? d)) * t),
  };
}

// ============================================================ render por frame
let lastFrameTime = performance.now();
const prevPos = new THREE.Vector2();

function renderCar(now, dt) {
  const { pose } = sim;
  ego.group.position.copy(toWorld(pose.x, pose.y));
  ego.group.rotation.y = pose.heading;
  ego.steerPivots.forEach((p) => (p.rotation.y = pose.steer));

  const moved = Math.hypot(pose.x - prevPos.x, pose.y - prevPos.y);
  const speed = sim.frame?.car.speed ?? 0;
  ego.wheels.forEach((w) => (w.rotation.z -= Math.sign(speed) * moved / WHEEL_RADIUS));
  prevPos.set(pose.x, pose.y);
  if (sim.showTrail && moved > 0) pushTrail(pose.x, pose.y);

  arrowHolder.rotation.y = speed < -0.05 ? Math.PI : 0;
  arrowHolder.position.y = 2.0 + Math.sin(now / 300) * 0.06;

  // Indicador de colisión: anillo pulsante y carrocería en rojo
  const crashed = sim.frame?.status === "collision";
  const pulse = crashed ? 0.5 + 0.5 * Math.sin((now - sim.statusAt) / 90) : 0;
  collisionRing.material.opacity = crashed ? 0.4 + 0.5 * pulse : 0;
  collisionRing.scale.setScalar(crashed ? 1 + 0.15 * pulse : 1);
  ego.bodyMat.emissive.setHex(crashed ? PALETTE.collision : 0x000000);
  ego.bodyMat.emissiveIntensity = crashed ? 0.6 * pulse : 0;
  egoGlow.material.color.setHex(crashed ? PALETTE.collision : PALETTE.ego);

  // Sensores
  const [hl, hw] = sim.carHalf;
  const range = sim.frame?.sensor_range ?? 8;
  sensorGroup.visible = sim.showSensors;
  sim.frame?.sensors.forEach((s, i) => {
    const sensor = sensors[i];
    const c = Math.cos(s.angle), sn = Math.sin(s.angle);
    const body = Math.min(Math.abs(c) > 1e-6 ? hl / Math.abs(c) : Infinity, Math.abs(sn) > 1e-6 ? hw / Math.abs(sn) : Infinity);
    const dist = Math.max(pose.sensors[i] ?? s.distance, 0.02);
    sensor.beam.position.set(c * body, 0, -sn * body);
    sensor.beam.rotation.y = s.angle;
    sensor.beam.scale.x = dist;
    sensor.hit.position.set(c * (body + dist), 0, -sn * (body + dist));
    sensor.hit.visible = dist < range - 0.05;
    const danger = THREE.MathUtils.clamp(1 - (dist - 0.5) / 2, 0, 1);
    sensor.material.color.setHex(PALETTE.sensor).lerp(new THREE.Color(PALETTE.collision), danger);
    sensor.material.opacity = 0.35 + 0.5 * (1 - dist / range);
  });

  // Cupo objetivo: pulso suave, más intenso al estacionar
  const success = sim.frame?.status === "success";
  targetFillMat.opacity = success ? 0.55 + 0.1 * Math.sin(now / 120) : 0.18 + 0.08 * Math.sin(now / 400);
  pin.position.y = 3.2 + Math.sin(now / 500) * 0.15;
  targetBeam.material.opacity = success ? 1 : 0.7;
}

const cameraTween = { active: false, start: 0, fromPos: new THREE.Vector3(), fromTarget: new THREE.Vector3(),
  toPos: new THREE.Vector3(), toTarget: new THREE.Vector3() };

function setCamera(mode) {
  sim.camera = mode;
  document.querySelectorAll("#camera-select button").forEach((b) => b.classList.toggle("active", b.dataset.camera === mode));
  $("hud-camera").textContent = `Cámara: ${CAMERA_LABELS[mode]}`;
  controls.enabled = false;
  if (mode === "follow") return;
  Object.assign(cameraTween, { active: true, start: performance.now() });
  cameraTween.fromPos.copy(camera.position);
  cameraTween.fromTarget.copy(controls.target);
  if (mode === "orbit") {
    cameraTween.toPos.set(0, 22, 25);
    cameraTween.toTarget.set(0, 0, 0);
  } else {
    cameraTween.toPos.set(0, 38, 0.01);
    cameraTween.toTarget.set(0, 0, 0);
  }
}

function updateCamera(dt) {
  if (sim.camera === "follow") {
    const { x, y, heading } = sim.pose;
    const back = toWorld(x - Math.cos(heading) * 8, y - Math.sin(heading) * 8, 4.5);
    const look = toWorld(x + Math.cos(heading) * 2.5, y + Math.sin(heading) * 2.5, 0.6);
    const k = 1 - Math.exp(-dt * 3.5);
    camera.position.lerp(back, k);
    controls.target.lerp(look, k);
    camera.lookAt(controls.target);
    return;
  }
  if (cameraTween.active) {
    const t = Math.min(1, (performance.now() - cameraTween.start) / 900);
    const e = 1 - Math.pow(1 - t, 3);
    camera.position.lerpVectors(cameraTween.fromPos, cameraTween.toPos, e);
    controls.target.lerpVectors(cameraTween.fromTarget, cameraTween.toTarget, e);
    camera.lookAt(controls.target);
    if (t >= 1) {
      cameraTween.active = false;
      controls.enabled = sim.camera === "orbit";
    }
    return;
  }
  if (sim.camera === "orbit") controls.update();
}

function resize() {
  const { clientWidth: w, clientHeight: h } = viewport;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);

function tick(now) {
  const dt = Math.min(0.1, (now - lastFrameTime) / 1000);
  lastFrameTime = now;
  if (sim.running && !sim.inFlight) {
    if (sim.episodeOver && now >= sim.autoResetAt) doReset();
    else if (!sim.episodeOver && now - sim.lastStepAt >= stepInterval()) doStep();
  }
  updatePose(now);
  renderCar(now, dt);
  updateCamera(dt);
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

// ============================================================ UI
const connection = new Connection((ok) => {
  $("conn").classList.toggle("ok", ok);
  $("conn").title = ok ? "Conectado" : "Desconectado";
  if (!ok) $("loading").textContent = "Sin conexión con el servidor. Reintentando…";
});

function showError(err) {
  const el = $("error");
  el.textContent = err.message ?? String(err);
  el.hidden = false;
  clearTimeout(showError.timer);
  showError.timer = setTimeout(() => (el.hidden = true), 4500);
}

function showBanner(status) {
  const text = { success: "✓ ¡Estacionado correctamente!", collision: "✕ Colisión detectada", timeout: "⏱ Tiempo agotado" }[status];
  const el = $("banner");
  el.textContent = text;
  el.className = `banner ${status}`;
  el.hidden = false;
  const flash = $("flash");
  flash.style.setProperty("--flash", status === "success" ? "rgba(34,197,94,0.5)" : status === "collision" ? "rgba(239,68,68,0.6)" : "rgba(245,158,11,0.4)");
  flash.classList.add("on");
  setTimeout(() => flash.classList.remove("on"), 700);
}

function hideBanner() {
  $("banner").hidden = true;
}

function setText(id, text) {
  $(id).textContent = text;
}

function updatePanel(frame) {
  let status = frame.status;
  if (status === "running" && !sim.running) status = frame.steps === 0 ? "idle" : "paused";
  const statusEl = $("status");
  statusEl.dataset.state = status;
  statusEl.textContent = STATUS_LABELS[status];
  setText("m-episode", frame.episode);
  setText("m-step", frame.steps);
  const rewardEl = $("m-reward");
  rewardEl.textContent = frame.reward.toFixed(2);
  rewardEl.className = frame.reward >= 0 ? "positive" : "negative";
  setText("m-total", `total ${frame.total_reward.toFixed(1)}`);
  setText("m-distance", `${frame.distance.toFixed(2)} m`);
  setText("m-heading", `orientación ${(frame.heading_error * 180 / Math.PI).toFixed(0)}°`);
  setText("m-speed", `${(frame.car.speed * 3.6).toFixed(1)} km/h`);
  setText("m-steer", `dirección ${(frame.car.steer * 180 / Math.PI).toFixed(0)}°`);
  setText("m-collisions", frame.collisions_total);
  setText("m-successes", `éxitos ${frame.successes_total}`);
  setText("m-agent", AGENT_LABELS[frame.agent]);
  setText("m-trained", frame.agent_trained ? "modelo cargado" : "sin entrenar · usa Entrenar");
  setText("hud-action", `Acción: ${frame.action_name ?? "—"}`);
  document.querySelectorAll("#agent-select button").forEach((b) => b.classList.toggle("active", b.dataset.agent === frame.agent));
  $("btn-train").disabled = frame.agent === "random" || frame.training.running;
}

let wasTraining = false;
function updateTraining(training) {
  const bar = $("train-bar");
  if (training.running || training.total) {
    bar.style.width = `${(100 * training.episode) / Math.max(1, training.total)}%`;
    setText("train-label", training.running
      ? `${AGENT_LABELS[training.agent]} · episodio ${training.episode}/${training.total}`
      : `${AGENT_LABELS[training.agent]} · completado (${training.total} episodios)`);
    setText("train-success", `éxito ${(training.success_rate * 100).toFixed(0)}%`);
  }
  if (training.error) showError(new Error(`Entrenamiento: ${training.error}`));
  drawSparkline(training.returns ?? []);
  if (wasTraining && !training.running) refreshConfig();
  wasTraining = training.running;
}

function drawSparkline(values) {
  const canvas = $("sparkline");
  const ratio = devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * ratio;
  canvas.height = h * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, w, h);
  if (values.length < 2) return;
  const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
  const y = (v) => h - 4 - ((v - min) / span) * (h - 8);
  if (min < 0 && max > 0) {
    ctx.strokeStyle = "rgba(139,147,167,0.3)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(0, y(0)); ctx.lineTo(w, y(0)); ctx.stroke();
    ctx.setLineDash([]);
  }
  const gradient = ctx.createLinearGradient(0, 0, w, 0);
  gradient.addColorStop(0, "#38bdf8");
  gradient.addColorStop(1, "#22c55e");
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = (i / (values.length - 1)) * w;
    i ? ctx.lineTo(x, y(v)) : ctx.moveTo(x, y(v));
  });
  ctx.stroke();
}

async function refreshConfig() {
  const config = await connection.send("config");
  document.querySelectorAll("#agent-select button").forEach((b) => {
    b.classList.toggle("untrained", !config.trained[b.dataset.agent]);
    b.title = config.trained[b.dataset.agent] ? "" : "Sin entrenar";
  });
  $("difficulty").value = config.difficulty;
  return config;
}

async function doStep() {
  sim.inFlight = true;
  try {
    applyFrame(await connection.send("step"));
  } catch (err) {
    showError(err);
    sim.running = false;
  } finally {
    sim.lastStepAt = performance.now();
    sim.inFlight = false;
  }
}

async function doReset(payload = {}) {
  sim.inFlight = true;
  try {
    applyReset(await connection.send("reset", payload));
  } catch (err) {
    showError(err);
  } finally {
    sim.lastStepAt = performance.now();
    sim.inFlight = false;
  }
}

function setRunning(running) {
  sim.running = running;
  $("btn-start").disabled = running;
  $("btn-pause").disabled = !running;
  if (sim.frame) updatePanel(sim.frame);
}

$("btn-start").onclick = () => {
  if (sim.episodeOver) doReset();
  setRunning(true);
};
$("btn-pause").onclick = () => setRunning(false);
$("btn-reset").onclick = () => doReset();

document.querySelectorAll("#agent-select button").forEach((button) => {
  button.onclick = async () => {
    try {
      await connection.send("agent", { agent: button.dataset.agent });
      if (sim.frame) updatePanel({ ...sim.frame, agent: button.dataset.agent,
        agent_trained: !button.classList.contains("untrained") });
      await doReset();
    } catch (err) { showError(err); }
  };
});
document.querySelectorAll("#camera-select button").forEach((b) => (b.onclick = () => setCamera(b.dataset.camera)));

$("speed").oninput = (e) => {
  sim.speed = Number(e.target.value);
  setText("speed-label", `${sim.speed.toFixed(2).replace(/0$/, "")}×`);
};
$("difficulty").onchange = (e) => doReset({ difficulty: e.target.value });
$("tg-sensors").onchange = (e) => (sim.showSensors = e.target.checked);
$("tg-trail").onchange = (e) => {
  sim.showTrail = e.target.checked;
  trailMesh.visible = sim.showTrail;
};
$("tg-noise").onchange = (e) => doReset({ sensor_noise: e.target.checked ? 0.1 : 0 });

$("btn-train").onclick = async () => {
  const episodes = Number($("train-episodes").value);
  if (!Number.isInteger(episodes) || episodes < 1 || episodes > 50000) {
    showError(new Error("Los episodios deben ser un entero entre 1 y 50000"));
    return;
  }
  try {
    const res = await connection.send("train", { agent: sim.frame.agent, episodes });
    updateTraining(res.training);
    $("btn-train").disabled = true;
  } catch (err) { showError(err); }
};

// Mientras se entrena con la simulación en pausa, se consulta el progreso periódicamente
setInterval(async () => {
  if (!sim.frame?.training.running || sim.running) return;
  try {
    const state = await connection.send("state");
    sim.frame.training = state.training;
    updateTraining(state.training);
  } catch { /* la reconexión se maneja en Connection */ }
}, 600);

window.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  const key = e.key.toLowerCase();
  if (key === " ") { e.preventDefault(); sim.running ? setRunning(false) : $("btn-start").click(); }
  else if (key === "r") doReset();
  else if (key === "c") setCamera({ orbit: "top", top: "follow", follow: "orbit" }[sim.camera]);
  else if (key === "s") $("tg-sensors").click();
  else if (key === "t") $("tg-trail").click();
});

// ============================================================ inicio
(async () => {
  resize();
  setRunning(false);
  setCamera("orbit");
  requestAnimationFrame(tick);
  try {
    const config = await refreshConfig();
    setText("m-agent", AGENT_LABELS[config.agent]);
    await doReset();
    $("loading").hidden = true;
  } catch (err) {
    showError(err);
  }
})();
