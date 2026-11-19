import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { Reflector } from "three/addons/objects/Reflector.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { themeFor } from "./themes";
import type {
  CatalogItem, Commands, Config, Frame, Layout, LevelId, Metrics, ResetFrame, Slot, Status, Training,
} from "./types";

// Dirección del servidor de la simulación. En local queda vacía (la web la sirve el mismo servidor);
// en la versión publicada (Vercel) la fija VITE_API_BASE_URL al construir.
const API_BASE = (import.meta.env?.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
// Registro de experimentos (MLflow) publicado; en local, el `mlflow ui` en el puerto 5000
const MLFLOW_URL = (import.meta.env?.VITE_MLFLOW_URL ?? "http://127.0.0.1:5000").replace(/\/$/, "");
const MLFLOW_EXPERIMENTS: Record<string, number> = { "parking-N1": 1, "parking-N2": 2, "parking-N3": 3, "parking-N4": 4, "parking-N5": 5 };
// El nivel va en el link (?level=N3): así cada visitante tiene su partida y la recarga lo conserva
let currentLevel: LevelId | null = new URLSearchParams(location.search).get("level");

// El servidor gratuito se duerme tras un rato sin visitas: si tarda, se avisa en lugar de parecer colgado
const wakeNotice = setTimeout(() => {
  const el = document.getElementById("loading");
  if (el && !el.hidden && el.lastChild) el.lastChild.textContent = "Despertando el servidor… la primera visita puede tardar hasta 1 minuto";
}, 4000);

// La paleta depende del nivel; al cambiar a un nivel con otra paleta la página se recarga
const THEME = themeFor(await fetch(`${API_BASE}/api/config${currentLevel ? `?level=${currentLevel}` : ""}`)
  .then((r) => r.json() as Promise<Config>).then((c) => (currentLevel = c.level)).catch(() => currentLevel ?? "N1"));
document.documentElement.dataset.theme = THEME.id;

// ============================================================ constantes
const PALETTE = {
  ego: THEME.ego,
  success: 0x22c55e,
  collision: 0xef4444,
  sensor: 0x38bdf8,
  parked: THEME.parked,
};
const STATUS_LABELS: Record<Status, string> = {
  idle: "En espera", running: "En curso", paused: "Pausado",
  success: "Estacionado", collision: "Colisión", timeout: "Tiempo agotado",
};
const TITLE_LABELS: Record<Status, string> = {
  idle: "Parquea el auto", running: "Parquea el auto", paused: "En pausa",
  success: "¡Estacionado!", collision: "¡Choque!", timeout: "Tiempo agotado",
};
const COLLISION_LABELS: Record<string, string> = {
  pedestrian: "¡Atropellaste a un peatón!", obstacle: "¡Choque con un obstáculo!", boundary: "¡Te saliste del parqueadero!",
};
const COLLISION_CAUSES: Record<string, string> = { pedestrian: "peatón", obstacle: "obstáculo", boundary: "salida del área" };
const BASE_STEP_MS = 200;      // dt del ambiente a velocidad 1×
const AUTO_RESET_MS = 1800;
const BRAKE_ACTION = 4;
const CEILING_H = 5.6;

/** Elemento de la página por id (la interfaz se arma en index.html: si falta, es un error de programación). */
function $<T extends HTMLElement = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Falta el elemento #${id} en index.html`);
  return el as T;
}
const THEMES_DIFFER = (level: LevelId) => themeFor(level).id !== THEME.id;

// ============================================================ conexión
type Pending = { resolve: (value: unknown) => void; reject: (err: Error) => void };

class Connection {
  private ws!: WebSocket;
  private ready!: Promise<void>;
  private pending: Pending[] = [];

  constructor(private onStatus: (ok: boolean) => void) {
    this.connect();
  }

  connect() {
    const base = API_BASE ? new URL(API_BASE) : location;
    const proto = base.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${base.host}/ws${currentLevel ? `?level=${currentLevel}` : ""}`);
    this.ready = new Promise<void>((resolve) => (this.ws.onopen = () => { this.onStatus(true); resolve(); }));
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
  async send<K extends keyof Commands>(cmd: K, payload: Commands[K]["payload"] = {} as Commands[K]["payload"]):
    Promise<Commands[K]["response"]> {
    await this.ready;
    return new Promise((resolve, reject) => {
      this.pending.push({ resolve: resolve as (value: unknown) => void, reject });
      this.ws.send(JSON.stringify({ cmd, payload }));
    });
  }
}

// ============================================================ renderer, cámara y luces
const viewport = $("viewport");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;
viewport.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(THEME.background);
scene.fog = new THREE.Fog(THEME.fog, 45, 95);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 300);
camera.position.set(0, 10, 15.5);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI / 2.08;
controls.minDistance = 5;
controls.maxDistance = 55;

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.45, 0.4, 1.15);
composer.addPass(bloom);
composer.addPass(new OutputPass());

scene.add(new THREE.HemisphereLight(THEME.hemiSky, THEME.hemiGround, 0.8));
const keyLight = new THREE.DirectionalLight(THEME.keyLight, 1.1);
keyLight.position.set(6, 22, 9);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
Object.assign(keyLight.shadow.camera, { left: -22, right: 22, top: 15, bottom: -15, near: 1, far: 60 });
keyLight.shadow.bias = -0.0004;
keyLight.shadow.normalBias = 0.03;
scene.add(keyLight);

// env (x, y) del simulador → (x, 0, -y) en Three.js; el rumbo θ es rotation.y
const toWorld = (x: number, y: number, h = 0) => new THREE.Vector3(x, h, -y);

function std(params: THREE.MeshStandardMaterialParameters) {
  return new THREE.MeshStandardMaterial({ envMapIntensity: 0.35, ...params });
}

// Material que supera el umbral del bloom (neón, lámparas, ventanas)
function glow(color: THREE.ColorRepresentation, intensity = 1, params: THREE.MeshBasicMaterialParameters = {}) {
  const material = new THREE.MeshBasicMaterial({ color, toneMapped: false, ...params });
  material.color.multiplyScalar(intensity);
  return material;
}

function shadowed<T extends THREE.Object3D>(mesh: T, cast = true, receive = true): T {
  mesh.castShadow = cast;
  mesh.receiveShadow = receive;
  return mesh;
}

function canvasTexture(width: number, height: number,
                       draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  draw(canvas.getContext("2d")!, width, height);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

// Semilla fija para que el decorado sea siempre el mismo
let decorSeed = 7;
const rand = () => ((decorSeed = (decorSeed * 16807) % 2147483647) / 2147483647);

// ============================================================ texturas procedurales
function floorTexture() {
  const texture = canvasTexture(1024, 1024, (ctx, w, h) => {
    ctx.fillStyle = THEME.floor;
    ctx.fillRect(0, 0, w, h);
    const image = ctx.getImageData(0, 0, w, h);
    for (let i = 0; i < image.data.length; i += 4) {
      const n = (Math.random() - 0.5) * 14;
      image.data[i] += n; image.data[i + 1] += n; image.data[i + 2] += n;
    }
    ctx.putImageData(image, 0, 0);
    ctx.strokeStyle = THEME.floorJoint;   // juntas del concreto pulido
    ctx.lineWidth = 2;
    for (let i = 0; i <= 4; i++) {
      ctx.beginPath(); ctx.moveTo((i * w) / 4, 0); ctx.lineTo((i * w) / 4, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, (i * h) / 4); ctx.lineTo(w, (i * h) / 4); ctx.stroke();
    }
  });
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(4, 3);
  return texture;
}

function hazardTexture(repeat: number) {
  const texture = canvasTexture(128, 64, (ctx, w, h) => {
    ctx.fillStyle = "#f5c518";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#26232e";
    for (let i = -h; i < w + h; i += 64) {
      ctx.beginPath();
      ctx.moveTo(i, h); ctx.lineTo(i + 32, h); ctx.lineTo(i + 32 + h, 0); ctx.lineTo(i + h, 0);
      ctx.fill();
    }
  });
  texture.wrapS = THREE.RepeatWrapping;
  texture.repeat.set(repeat, 1);
  return texture;
}

function pillarTexture() {
  return canvasTexture(128, 512, (ctx, w, h) => {
    const [a, b] = THEME.pillarBands;
    const bands: [number, number, string][] = [[0, 0.3, a], [0.3, 0.5, b], [0.5, 0.82, a], [0.82, 1, b]];
    for (const [a, b, color] of bands) {
      ctx.fillStyle = color;
      ctx.fillRect(0, a * h, w, (b - a) * h + 1);
    }
    ctx.fillStyle = "rgba(255,255,255,0.18)";
    ctx.fillRect(0, 0, w * 0.12, h);
    ctx.fillStyle = "#ffffff";
    ctx.font = "700 46px Fredoka, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("P1", w / 2, 0.4 * h);
  });
}

function skylineTexture() {
  return canvasTexture(1024, 384, (ctx, w, h) => {
    const sky = ctx.createLinearGradient(0, 0, 0, h);
    [0, 0.45, 0.75, 1].forEach((stop, i) => sky.addColorStop(stop, THEME.sky[i]));
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, h);
    for (let layer = 0; layer < 2; layer++) {
      let x = -20;
      while (x < w) {
        const bw = 40 + rand() * 70, bh = (layer ? 120 : 70) + rand() * (layer ? 170 : 120);
        ctx.fillStyle = THEME.buildings[layer];
        ctx.fillRect(x, h - bh, bw, bh);
        if (layer) {
          for (let wy = h - bh + 10; wy < h - 10; wy += 14) {
            for (let wx = x + 6; wx < x + bw - 8; wx += 11) {
              if (rand() < 0.45) {
                ctx.fillStyle = THEME.windows[rand() < 0.8 ? 0 : 1];
                ctx.fillRect(wx, wy, 5, 7);
              }
            }
          }
        }
        x += bw + (layer ? 6 : -10);
      }
    }
    // Marco y parteluces de la ventana
    ctx.strokeStyle = THEME.windowFrame;
    ctx.lineWidth = 14;
    ctx.strokeRect(0, 0, w, h);
    ctx.lineWidth = 8;
    for (let i = 1; i < 4; i++) {
      ctx.beginPath(); ctx.moveTo((i * w) / 4, 0); ctx.lineTo((i * w) / 4, h); ctx.stroke();
    }
  });
}

function neonTexture() {
  return canvasTexture(1024, 256, (ctx, w, h) => {
    ctx.strokeStyle = THEME.neon.stroke;
    ctx.fillStyle = THEME.neon.fill;
    ctx.shadowColor = THEME.neon.shadow;
    ctx.shadowBlur = 28;
    ctx.lineWidth = 10;
    ctx.beginPath();
    ctx.arc(128, h / 2, 88, 0, Math.PI * 2);
    ctx.stroke();
    ctx.font = "700 130px Fredoka, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let i = 0; i < 2; i++) ctx.fillText("P", 128, h / 2 + 8);
    ctx.font = "600 150px Fredoka, sans-serif";
    ctx.textAlign = "left";
    for (let i = 0; i < 2; i++) ctx.fillText("PARKING", 260, h / 2 + 10);
  });
}

function targetTexture(aspect: number) {
  const W = 256, H = Math.round(256 * aspect);
  return canvasTexture(W, H, (ctx, w, h) => {
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 12;
    ctx.setLineDash([30, 18]);
    ctx.beginPath();
    ctx.roundRect(10, 10, w - 20, h - 20, 22);
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.font = `700 ${w * 0.62}px Fredoka, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("P", w / 2, h / 2 + w * 0.04);
  });
}

function labelTexture(text: string, color: string = THEME.label) {
  return canvasTexture(256, 128, (ctx, w, h) => {
    ctx.fillStyle = color;
    ctx.font = "600 72px Fredoka, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, w / 2, h / 2 + 4);
  });
}

function flatPlane(width: number, depth: number, material: THREE.Material, x: number, y: number, height = 0.01) {
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, depth), material);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.copy(toWorld(x, y, height));
  mesh.receiveShadow = true;
  return mesh;
}

type Vec3 = [number, number, number];
function boxPart(material: THREE.Material, size: Vec3, position: Vec3, geometry: THREE.BufferGeometry = unitBox) {
  const mesh = shadowed(new THREE.Mesh(geometry, material));
  mesh.scale.set(...size);
  mesh.position.set(...position);
  return mesh;
}
const unitBox = new THREE.BoxGeometry(1, 1, 1);

// ============================================================ garaje estático
const paint = std({ color: 0xffffff, roughness: 0.5, emissive: THEME.paintEmissive });
const pillarMaterials = (() => {
  const side = std({ map: pillarTexture(), roughness: 0.6 });
  const cap = std({ color: THEME.pillarCap, roughness: 0.6 });
  return [side, side, cap, cap, side, side];
})();
const ceilingGroup = new THREE.Group();
let garageBuilt = false;

function addPillar(group: THREE.Group, x: number, y: number, size = 0.8) {
  const pillar = shadowed(new THREE.Mesh(unitBox, pillarMaterials));
  pillar.scale.set(size, CEILING_H, size);
  pillar.position.copy(toWorld(x, y, CEILING_H / 2));
  group.add(pillar);
}

function addPlanter(group: THREE.Group, x: number, y: number) {
  const pot = std({ color: 0xf4f1fb, roughness: 0.4 });
  const leaf = std({ color: 0x3f9b5c, roughness: 0.6, side: THREE.DoubleSide });
  const leafDark = std({ color: 0x2d7a47, roughness: 0.6, side: THREE.DoubleSide });
  const planter = new THREE.Group();
  planter.add(boxPart(pot, [0.9, 0.7, 0.9], [0, 0.35, 0]));
  planter.add(boxPart(std({ color: 0x4a3a33, roughness: 1 }), [0.8, 0.05, 0.8], [0, 0.7, 0]));
  const leafGeo = new THREE.SphereGeometry(1, 10, 8);
  for (let i = 0; i < 14; i++) {
    const a = (i / 14) * Math.PI * 2 + rand() * 0.4;
    const tilt = 0.5 + rand() * 0.6;
    const len = 0.55 + rand() * 0.35;
    const l = shadowed(new THREE.Mesh(leafGeo, i % 2 ? leaf : leafDark), true, false);
    l.scale.set(0.1, len, 0.035);
    l.position.set(Math.cos(a) * 0.25, 0.72 + len * 0.8, Math.sin(a) * 0.25);
    l.rotation.set(0, -a, 0);
    l.rotateZ(-tilt);
    planter.add(l);
  }
  planter.position.copy(toWorld(x, y));
  group.add(planter);
}

function buildGarage(layout: Layout) {
  const { width: W, height: H } = layout.lot;
  const GX = W / 2 + 3.2, GY = H / 2 + 1.3;    // paredes del edificio
  const group = new THREE.Group();

  // Piso brillante: espejo debajo de una capa de concreto semitransparente
  const mirror = new Reflector(new THREE.PlaneGeometry(2 * GX, 2 * GY), {
    textureWidth: 1024, textureHeight: 1024, color: THEME.mirror, clipBias: 0.003,
  });
  mirror.rotation.x = -Math.PI / 2;
  group.add(mirror);
  const floor = flatPlane(2 * GX, 2 * GY,
    std({ map: floorTexture(), roughness: 0.4, transparent: true, opacity: 0.84 }), 0, 0, 0.003);
  floor.renderOrder = -1;
  group.add(floor);

  // Paredes orientadas hacia adentro: desde afuera son invisibles (vista de "maqueta")
  const wallMat = std({ color: THEME.wall, roughness: 0.85 });
  const baseMat = std({ color: THEME.wallBase, roughness: 0.8 });
  const skyline = glow(0xffffff, 1, { map: skylineTexture() });
  const walls = [
    { x: 0, y: GY, rot: 0, len: 2 * GX },
    { x: 0, y: -GY, rot: Math.PI, len: 2 * GX },
    { x: GX, y: 0, rot: -Math.PI / 2, len: 2 * GY },
    { x: -GX, y: 0, rot: Math.PI / 2, len: 2 * GY },
  ];
  for (const wall of walls) {
    const holder = new THREE.Group();
    holder.position.copy(toWorld(wall.x, wall.y));
    holder.rotation.y = wall.rot;
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(wall.len, CEILING_H), wallMat);
    plane.position.y = CEILING_H / 2;
    plane.receiveShadow = true;
    const base = new THREE.Mesh(new THREE.PlaneGeometry(wall.len, 0.35), baseMat);
    base.position.set(0, 0.175, 0.01);
    holder.add(plane, base);
    const windows = wall.rot % Math.PI === 0
      ? [[-(wall.len / 2 - 4.2), 6.4], [wall.len / 2 - 4.2, 6.4]]
      : [[0, wall.len - 4]];
    for (const [cx, width] of windows) {
      const win = new THREE.Mesh(new THREE.PlaneGeometry(width, 2.9), skyline);
      win.position.set(cx, 2.75, 0.02);
      holder.add(win);
    }
    if (wall.rot % Math.PI === 0) {
      const neon = new THREE.Mesh(new THREE.PlaneGeometry(7.2, 1.8),
        glow(0xffffff, 1.7, { map: neonTexture(), transparent: true, depthWrite: false }));
      neon.position.set(0, 3.5, 0.04);
      holder.add(neon);
    }
    group.add(holder);
  }

  // Techo: losa, vigas, tuberías y lámparas lineales
  const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(2 * GX, 2 * GY), std({ color: THEME.ceiling, roughness: 0.9 }));
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.y = CEILING_H;
  ceilingGroup.add(ceiling);
  const beamMat = std({ color: THEME.beam, roughness: 0.8 });
  for (let x = -GX + 3; x < GX; x += 6) ceilingGroup.add(boxPart(beamMat, [0.45, 0.5, 2 * GY], [x, CEILING_H - 0.25, 0]));
  const pipeGeo = new THREE.CylinderGeometry(1, 1, 1, 12).rotateZ(Math.PI / 2);
  const pipeMats = [std({ color: THEME.pipes[0], roughness: 0.4, metalness: 0.3 }), std({ color: THEME.pipes[1], roughness: 0.4, metalness: 0.5 })];
  [[8.6, 0.1, 0], [9.0, 0.07, 1], [-9.2, 0.08, 0]].forEach(([y, r, m]) => {
    const pipe = new THREE.Mesh(pipeGeo, pipeMats[m]);
    pipe.scale.set(2 * GX, r, r);
    pipe.position.copy(toWorld(0, y, CEILING_H - 0.6 - r));
    ceilingGroup.add(pipe);
  });
  const lampMat = glow(THEME.lamp, 2.6);
  const lampHousing = std({ color: THEME.lampHousing, roughness: 0.5 });
  for (const y of [-4.2, 0, 4.2]) {
    for (let x = -12; x <= 12; x += 6) {
      ceilingGroup.add(boxPart(lampHousing, [2.8, 0.1, 0.42], [x, CEILING_H - 0.55, -y]));
      ceilingGroup.add(boxPart(lampMat, [2.6, 0.04, 0.26], [x, CEILING_H - 0.62, -y]));
    }
  }
  scene.add(ceilingGroup);
  for (const [x, y] of [[-10, 4], [0, 4], [10, 4], [-10, -4], [0, -4], [10, -4]]) {
    const light = new THREE.PointLight(THEME.pointLight, 14, 16, 1.5);
    light.position.copy(toWorld(x, y, CEILING_H - 0.9));
    group.add(light);
  }

  // Bordillos con franjas de peligro
  const addCurb = (x: number, y: number, length: number, alongX: boolean) => {
    const curb = shadowed(new THREE.Mesh(unitBox, std({ map: hazardTexture(length / 1.2), roughness: 0.6 })));
    curb.scale.set(length, 0.22, 0.34);
    curb.position.copy(toWorld(x, y, 0.11));
    if (!alongX) curb.rotation.y = Math.PI / 2;
    group.add(curb);
  };
  addCurb(0, H / 2 + 0.17, W + 0.68, true);
  addCurb(0, -H / 2 - 0.17, W + 0.68, true);
  addCurb(W / 2 + 0.17, 0, H, false);
  addCurb(-W / 2 - 0.17, 0, H, false);

  // Líneas de los cupos, topes y numeración
  const stopMat = std({ map: hazardTexture(2), roughness: 0.6 });
  const rows = [0, 1].map((r) => layout.slots.filter((s) => s.row === r));
  rows.forEach((slots, r) => {
    const { width: sw, depth: sd } = slots[0];
    const sign = r === 0 ? 1 : -1;
    const inner = sign * (H / 2 - sd), outer = sign * (H / 2);
    const xs = slots.map((s) => s.x);
    for (let i = 0; i <= slots.length; i++) {
      const x = Math.min(...xs) - sw / 2 + i * sw;
      group.add(flatPlane(0.14, sd, paint, x, (inner + outer) / 2, 0.012));
    }
    group.add(flatPlane(slots.length * sw, 0.14, paint, 0, inner, 0.012));
    slots.forEach((slot, i) => {
      const stop = shadowed(new THREE.Mesh(unitBox, stopMat));
      stop.scale.set(1.5, 0.12, 0.2);
      stop.position.copy(toWorld(slot.x, outer - sign * 0.55, 0.06));
      group.add(stop);
      const label = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 0.6),
        new THREE.MeshBasicMaterial({ map: labelTexture(`${"AB"[r]}${i + 1}`), transparent: true, opacity: 0.9, depthWrite: false }));
      label.rotation.set(-Math.PI / 2, 0, r === 0 ? 0 : Math.PI);
      label.position.copy(toWorld(slot.x, outer - sign * 1.2, 0.014));
      group.add(label);
    });
  });

  // Pasillo: línea discontinua y flechas
  for (let x = -W / 2 + 2; x < W / 2 - 2; x += 2.2) group.add(flatPlane(1.2, 0.16, paint, x + 0.6, 0, 0.012));
  const arrowShape = new THREE.Shape();
  arrowShape.moveTo(-1.0, -0.2); arrowShape.lineTo(0.2, -0.2); arrowShape.lineTo(0.2, -0.55);
  arrowShape.lineTo(1.1, 0); arrowShape.lineTo(0.2, 0.55); arrowShape.lineTo(0.2, 0.2);
  arrowShape.lineTo(-1.0, 0.2); arrowShape.closePath();
  const arrowGeo = new THREE.ShapeGeometry(arrowShape);
  for (const x of [-9, 0, 9]) {
    for (const [lane, dir] of [[2, Math.PI], [-2, 0]]) {
      const arrow = new THREE.Mesh(arrowGeo, paint);
      arrow.rotation.set(-Math.PI / 2, 0, dir);
      arrow.position.copy(toWorld(x, lane, 0.013));
      arrow.receiveShadow = true;
      group.add(arrow);
    }
  }

  // Columnas decorativas y jardineras fuera del área de maniobra
  for (const sx of [-1, 1]) {
    for (const y of [-6, 0, 6]) addPillar(group, sx * (W / 2 + 1.6), y, 0.9);
    for (const sy of [-1, 1]) addPlanter(group, sx * (W / 2 + 1.6), sy * (GY - 0.7));
  }
  for (const x of [-5.5, 5.5]) for (const sy of [-1, 1]) addPlanter(group, x, sy * (GY - 0.55));

  scene.add(group);
  garageBuilt = true;
}


// ============================================================ garaje grande (N5): azotea de noche
// Mapa abierto con 3 filas, pasillos laterales y entrada abajo. Sin techo, sobre un fondo morado de noche.
function addLampPost(group: THREE.Group, x: number, y: number, light = true) {
  const pole = std({ color: 0x2a2440, roughness: 0.5, metalness: 0.6 });
  group.add(boxPart(pole, [0.14, 4.2, 0.14], [x, 2.1, -y]));
  group.add(boxPart(pole, [0.5, 0.12, 0.5], [x, 0.06, -y]));
  const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.26, 16, 12), glow(THEME.lamp, 3));
  bulb.position.set(x, 4.35, -y);
  group.add(bulb);
  if (light) {
    const l = new THREE.PointLight(THEME.pointLight, 22, 18, 1.4);
    l.position.set(x, 4.1, -y);
    group.add(l);
  }
}

function addTree(group: THREE.Group, x: number, y: number, scale = 1) {
  const trunk = std({ color: 0x5b4636, roughness: 0.9 });
  const leaves = [std({ color: 0x2f8f57, roughness: 0.7 }), std({ color: 0x23704a, roughness: 0.7 })];
  const tree = new THREE.Group();
  tree.add(boxPart(std({ color: 0xe9e4f5, roughness: 0.4 }), [1.2, 0.6, 1.2], [0, 0.3, 0]));
  tree.add(boxPart(trunk, [0.16, 1.4, 0.16], [0, 1.2, 0]));
  const geo = new THREE.IcosahedronGeometry(1, 1);
  for (let i = 0; i < 5; i++) {
    const blob = shadowed(new THREE.Mesh(geo, leaves[i % 2]), true, false);
    const r = 0.45 + rand() * 0.25;
    blob.scale.setScalar(r);
    blob.position.set((rand() - 0.5) * 0.8, 1.9 + rand() * 0.7, (rand() - 0.5) * 0.8);
    tree.add(blob);
  }
  tree.scale.setScalar(scale);
  tree.position.copy(toWorld(x, y));
  group.add(tree);
}

function addGate(group: THREE.Group, x: number, y: number, side: number) {
  // Caseta con talanquera levantada a cada lado de la entrada
  const booth = std({ color: 0x2c2547, roughness: 0.5, metalness: 0.3 });
  group.add(boxPart(booth, [0.7, 1.5, 0.7], [x, 0.75, -y]));
  const cap = new THREE.Mesh(unitBox, glow(THEME.lamp, 2));
  cap.scale.set(0.5, 0.12, 0.5);
  cap.position.set(x, 1.56, -y);
  group.add(cap);
  const arm = shadowed(new THREE.Mesh(unitBox, std({ map: hazardTexture(6), roughness: 0.5 })));
  arm.scale.set(4.1, 0.12, 0.12);
  arm.geometry = unitBox;
  const pivot = new THREE.Group();
  pivot.position.set(x, 1.25, -y);
  pivot.rotation.z = 0;                                // bajada: la entrada se cerró detrás del auto
  arm.position.set(-side * 2.05, 0, 0);
  pivot.add(arm);
  group.add(pivot);
}

function buildRooftop(layout: Layout) {
  const { width: W, height: H } = layout.lot;
  const M = 4;                                         // franja alrededor del área de maniobra
  const group = new THREE.Group();

  const mirror = new Reflector(new THREE.PlaneGeometry(W + 2 * M, H + 2 * M), {
    textureWidth: 1024, textureHeight: 1024, color: THEME.mirror, clipBias: 0.003,
  });
  mirror.rotation.x = -Math.PI / 2;
  group.add(mirror);
  const floorTex = floorTexture();
  floorTex.repeat.set(6, 6);
  const floor = flatPlane(W + 2 * M, H + 2 * M,
    std({ map: floorTex, roughness: 0.55, transparent: true, opacity: 0.86 }), 0, 0, 0.003);
  floor.renderOrder = -1;
  group.add(floor);

  // Baranda baja con remate luminoso justo en el borde del área (el choque con el borde es contra ella).
  // En la entrada (abajo, |x| < 4.2) las talanqueras quedan cerradas detrás del auto.
  const wallMat = std({ color: THEME.wall, roughness: 0.6 });
  const capMat = glow(THEME.neon.stroke, 1.6);
  const gx = W / 2 + 0.25, gy = H / 2 + 0.25;
  const parapet = (x: number, y: number, lx: number, ly: number) => {
    group.add(boxPart(wallMat, [lx, 1.0, ly], [x, 0.5, -y]));
    group.add(boxPart(capMat, [lx + 0.04, 0.06, ly + 0.04], [x, 1.03, -y]));
  };
  parapet(0, gy, 2 * gx, 0.5);
  parapet(gx, 0, 0.5, 2 * gy);
  parapet(-gx, 0, 0.5, 2 * gy);
  const doorHalf = 4.2;
  parapet(-(gx + doorHalf) / 2, -gy, gx - doorHalf, 0.5);
  parapet((gx + doorHalf) / 2, -gy, gx - doorHalf, 0.5);
  // Faroles, árboles y jardineras por fuera de la baranda (no estorban la maniobra)
  for (let x = -gx + 2; x <= gx - 2; x += 6) {
    addLampPost(group, x, gy + 0.7, Math.abs(x) < 10);
    if (Math.abs(x) > doorHalf + 1.5) addLampPost(group, x, -gy - 0.7, false);
  }
  for (let y = -gy + 4; y <= gy - 4; y += 8) {
    addLampPost(group, gx + 0.7, y, Math.abs(y) < 6);
    addLampPost(group, -gx - 0.7, y, Math.abs(y) < 6);
  }
  for (const [x, y] of [[-gx - 2, gy + 2], [gx + 2, gy + 2], [-gx - 2, -gy - 2], [gx + 2, -gy - 2],
                        [-doorHalf - 3, -gy - 1.8], [doorHalf + 3, -gy - 1.8]]) addTree(group, x, y, 1.1);
  for (let x = -gx + 4; x < gx - 3; x += 3.2) {
    if (Math.abs(x) > 8) addPlanter(group, x, gy + 1.6);
  }
  addGate(group, -doorHalf, -gy, -1);
  addGate(group, doorHalf, -gy, 1);
  // Luz general de la azotea
  for (const [x, y] of [[-10, 15], [10, 15], [-10, 1], [10, 1], [-8, -14], [8, -14], [0, -21]]) {
    const light = new THREE.PointLight(THEME.pointLight, 30, 26, 1.3);
    light.position.copy(toWorld(x, y, 7));
    group.add(light);
  }

  // Bordillos detrás de las filas B y C: franja con luz
  const curbMat = std({ color: THEME.wallBase, roughness: 0.6 });
  for (const o of layout.obstacles.filter((o) => o.kind === "curb" && Math.abs(o.y) < H / 2 - 0.5 && Math.abs(o.x) < W / 2 - 0.5)) {
    group.add(boxPart(curbMat, [o.hx * 2, 0.3, o.hy * 2], [o.x, 0.15, -o.y]));
    group.add(boxPart(capMat, [o.hx * 2, 0.04, 0.08], [o.x, 0.32, -o.y]));
  }

  // Líneas de los cupos, topes y nombres (A1…C8)
  const stopMat = std({ map: hazardTexture(2), roughness: 0.6 });
  const rowIds = [...new Set(layout.slots.map((s) => s.row))];
  for (const r of rowIds) {
    const slots = layout.slots.filter((s) => s.row === r);
    const { width: sw, depth: sd, opening } = slots[0];
    const yc = slots[0].y, front = yc + opening * sd / 2, back = yc - opening * sd / 2;
    const xs = slots.map((s) => s.x);
    for (let i = 0; i <= slots.length; i++) group.add(flatPlane(0.14, sd, paint, Math.min(...xs) - sw / 2 + i * sw, yc, 0.012));
    group.add(flatPlane(slots.length * sw, 0.14, paint, (Math.min(...xs) + Math.max(...xs)) / 2, back, 0.012));
    group.add(flatPlane(slots.length * sw, 0.14, paint, (Math.min(...xs) + Math.max(...xs)) / 2, front, 0.012));
    slots.forEach((slot) => {
      const stop = shadowed(new THREE.Mesh(unitBox, stopMat));
      stop.scale.set(1.5, 0.12, 0.2);
      stop.position.copy(toWorld(slot.x, back + opening * 0.55, 0.06));
      group.add(stop);
      const label = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 0.6),
        new THREE.MeshBasicMaterial({ map: labelTexture(slot.name ?? `${slot.id + 1}`), transparent: true, opacity: 0.9, depthWrite: false }));
      label.rotation.set(-Math.PI / 2, 0, opening < 0 ? 0 : Math.PI);
      label.position.copy(toWorld(slot.x, front - opening * 0.7, 0.014));
      group.add(label);
    });
  }

  // Flechas de circulación: se sube por la derecha, se cruza por los pasillos y se baja por la izquierda
  const arrowShape = new THREE.Shape();
  arrowShape.moveTo(-1.0, -0.2); arrowShape.lineTo(0.2, -0.2); arrowShape.lineTo(0.2, -0.55);
  arrowShape.lineTo(1.1, 0); arrowShape.lineTo(0.2, 0.55); arrowShape.lineTo(0.2, 0.2);
  arrowShape.lineTo(-1.0, 0.2); arrowShape.closePath();
  const arrowGeo = new THREE.ShapeGeometry(arrowShape);
  const arrow = (x: number, y: number, dir: number) => {
    const a = new THREE.Mesh(arrowGeo, paint);
    a.rotation.set(-Math.PI / 2, 0, dir);
    a.position.copy(toWorld(x, y, 0.013));
    group.add(a);
  };
  // Centro de cada pasillo: entre el frente de su fila y lo que haya debajo (bordillo o fila siguiente)
  const aisleYs = rowIds.map((r) => {
    const slot = layout.slots.find((s) => s.row === r)!;     // rowIds sale de los mismos cupos
    return slot.y + slot.opening * slot.depth / 2 + slot.opening * 4.25;
  });
  for (const y of [-14, 0, 14]) { arrow(16, y, Math.PI / 2); arrow(-16, y, -Math.PI / 2); }
  for (const y of aisleYs) for (const x of [-7, 7]) arrow(x, y, Math.PI);
  const lastAisle = Math.min(...aisleYs);
  arrow(0, (lastAisle - 4.25 - H / 2) / 2, Math.PI / 2);
  // Línea punteada de la entrada
  for (let y = -H / 2 + 0.5; y < lastAisle - 4.25; y += 1.6) group.add(flatPlane(0.16, 0.9, paint, 0, y, 0.012));

  scene.add(group);
  garageBuilt = true;
  // Cámara, sombras y niebla a la escala del mapa grande
  CAMERA_PRESETS.orbit = { pos: [0, 40, 37], target: [0, 0, 3.5] };
  CAMERA_PRESETS.top = { pos: [0, 62, 0.01], target: [0, 0, 0] };
  controls.maxDistance = 120;
  scene.fog = new THREE.Fog(THEME.fog, 90, 190);
  camera.near = 0.5;
  camera.far = 400;
  camera.updateProjectionMatrix();
  Object.assign(keyLight.shadow.camera, { left: -34, right: 34, top: 34, bottom: -34, far: 120 });
  keyLight.position.set(10, 34, 14);
  keyLight.shadow.camera.updateProjectionMatrix();
  setCamera(sim.camera ?? "orbit");
}

// ============================================================ superdeportivos (modelo glTF)
// Modelo de los ejemplos de three.js (ver README para el crédito). Mira hacia −z y mide 4.52 m;
// se escala al largo del simulador y se gira para que su frente quede en +x.
const CAR_MODEL_URL = "models/sports_car.glb";
const MODEL_LENGTH = 4.52, MODEL_WHEEL_RADIUS = 0.36;
const CAR_SCALE = 4.2 / MODEL_LENGTH;
const WHEEL_RADIUS = MODEL_WHEEL_RADIUS * CAR_SCALE;
const HIDDEN_PARTS = ["steering_wheel", "carpet", "leather"];   // interior no visible tras el vidrio
const WHEEL_NAMES = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"];
const INTERIOR_MATERIALS = ["Interior_light", "Interior_dark", "Leather", "Leather_red", "Carpet"];
let carTemplate: THREE.Group | null = null;

const arrowGeometry = (() => {
  const shape = new THREE.Shape();
  shape.moveTo(0.7, 0); shape.lineTo(-0.15, 0.45); shape.lineTo(0.05, 0);
  shape.lineTo(-0.15, -0.45); shape.closePath();
  return new THREE.ShapeGeometry(shape);
})();

const sharedMats = {
  glass: new THREE.MeshPhysicalMaterial({ color: 0x0f0d18, metalness: 0.4, roughness: 0.05, clearcoat: 1,
    envMapIntensity: 0.9, transparent: true, opacity: 0.92 }),
  details: std({ color: 0x2a2933, metalness: 0.9, roughness: 0.3, envMapIntensity: 1 }),
  interior: std({ color: 0x15131c, roughness: 0.7 }),
  head: new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xf4f0ff, emissiveIntensity: 0.35 }),
  tail: new THREE.MeshStandardMaterial({ color: 0x8a1020, emissive: 0xff2a3a, emissiveIntensity: 0.35 }),
};
const paintCache = new Map<number, THREE.MeshPhysicalMaterial>();
function paintMaterial(color: number): THREE.MeshPhysicalMaterial {
  if (!paintCache.has(color)) {
    paintCache.set(color, new THREE.MeshPhysicalMaterial({
      color, metalness: 0.6, roughness: 0.25, clearcoat: 1, clearcoatRoughness: 0.05, envMapIntensity: 1.1,
    }));
  }
  return paintCache.get(color)!;
}

async function loadCarModel() {
  const draco = new DRACOLoader().setDecoderPath("https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/libs/draco/gltf/");
  const gltf = await new GLTFLoader().setDRACOLoader(draco).loadAsync(CAR_MODEL_URL);
  const model = gltf.scene;
  HIDDEN_PARTS.forEach((name) => model.getObjectByName(name)?.removeFromParent());
  model.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.castShadow = mesh.receiveShadow = true;
    // Acentos amarillos originales de la marca → gris oscuro; interior oscuro como vidrio polarizado
    const name = (mesh.material as THREE.Material).name;
    if (name === "Ferrari_Yellow") mesh.material = sharedMats.details;
    else if (INTERIOR_MATERIALS.includes(name)) mesh.material = sharedMats.interior;
  });
  for (const name of WHEEL_NAMES) {
    if (!model.getObjectByName(name)) throw new Error(`El modelo no tiene la rueda ${name}`);
  }
  model.rotation.y = -Math.PI / 2;
  model.scale.setScalar(CAR_SCALE);
  // Centrar la carrocería en el origen (el ambiente usa el centro del vehículo)
  const box = new THREE.Box3().setFromObject(model);
  const center = box.getCenter(new THREE.Vector3());
  model.position.set(-center.x, -box.min.y, -center.z);
  carTemplate = model;
}

interface CarParts {
  group: THREE.Group;
  bodyMat: THREE.MeshPhysicalMaterial;
  tailMat: THREE.MeshStandardMaterial;
  wheels: THREE.Object3D[];
  steerPivots: THREE.Group[];
}

function makeCar({ color, ego = false }: { color: number; ego?: boolean }): CarParts {
  if (!carTemplate) throw new Error("El modelo del auto aún no se ha cargado");
  const group = new THREE.Group();
  const model = carTemplate.clone(true);
  group.add(model);
  const bodyMat = ego ? paintMaterial(color).clone() : paintMaterial(color);
  const tailMat = ego ? sharedMats.tail.clone() : sharedMats.tail;
  const headMat = ego ? sharedMats.head.clone() : sharedMats.head;
  if (ego) {
    tailMat.emissiveIntensity = 1.2;
    headMat.emissiveIntensity = 2.4;
  }
  const setMaterial = (name: string, material: THREE.Material) => {
    model.getObjectByName(name)?.traverse((o) => { if ((o as THREE.Mesh).isMesh) (o as THREE.Mesh).material = material; });
  };
  setMaterial("body", bodyMat);
  setMaterial("glass", sharedMats.glass);
  setMaterial("trim", sharedMats.details);
  setMaterial("lights_red", tailMat);
  setMaterial("brakes", tailMat);
  setMaterial("lights", headMat);
  ["rim_fl", "rim_fr", "rim_rl", "rim_rr"].forEach((name) => setMaterial(name, sharedMats.details));

  // Las ruedas delanteras se montan en un pivote para poder girar la dirección
  const wheels: THREE.Object3D[] = [], steerPivots: THREE.Group[] = [];
  for (const name of WHEEL_NAMES) {
    const wheel = model.getObjectByName(name)!;          // loadCarModel ya comprobó que existen
    wheels.push(wheel);
    if (name.startsWith("wheel_f")) {
      const pivot = new THREE.Group();
      pivot.position.copy(wheel.position);
      wheel.parent!.add(pivot);
      wheel.position.set(0, 0, 0);
      pivot.add(wheel);
      steerPivots.push(pivot);
    }
  }
  return { group, bodyMat, wheels, steerPivots, tailMat };
}

// ============================================================ vehículo controlado
// El grupo existe desde el inicio; la carrocería se agrega cuando termina de cargar el modelo
const ego: CarParts = {
  group: new THREE.Group(), wheels: [], steerPivots: [],
  bodyMat: paintMaterial(PALETTE.ego).clone(), tailMat: sharedMats.tail.clone(),
};
scene.add(ego.group);

function mountEgoCar() {
  const car = makeCar({ color: PALETTE.ego, ego: true });
  ego.group.add(car.group);
  Object.assign(ego, car, { group: ego.group });
}

const egoArrow = new THREE.Mesh(arrowGeometry, glow(THEME.egoArrow, 1.2, { transparent: true, opacity: 0.95, side: THREE.DoubleSide }));
egoArrow.rotation.x = -Math.PI / 2;
const arrowHolder = new THREE.Group();
arrowHolder.position.y = 1.9;
arrowHolder.add(egoArrow);
ego.group.add(arrowHolder);

const egoGlow = new THREE.Mesh(new THREE.CircleGeometry(2.6, 48),
  new THREE.MeshBasicMaterial({ color: THEME.egoGlow, transparent: true, opacity: 0.16, depthWrite: false }));
egoGlow.rotation.x = -Math.PI / 2;
egoGlow.position.y = 0.025;
egoGlow.scale.set(1.12, 0.6, 1);
ego.group.add(egoGlow);

for (const z of [-0.58, 0.58]) {
  const spot = new THREE.SpotLight(THEME.headlights, 22, 14, 0.5, 0.6, 1.4);
  spot.position.set(2.0, 0.62, z);
  spot.target.position.set(8, 0, z * 2);
  ego.group.add(spot, spot.target);
}

const collisionRing = new THREE.Mesh(new THREE.RingGeometry(2.5, 2.8, 64),
  new THREE.MeshBasicMaterial({ color: PALETTE.collision, transparent: true, opacity: 0, depthWrite: false, toneMapped: false }));
collisionRing.rotation.x = -Math.PI / 2;
collisionRing.position.y = 0.03;
ego.group.add(collisionRing);

// Sensores: un haz y un punto de impacto por rayo, en el marco local del vehículo
const sensorGroup = new THREE.Group();
sensorGroup.position.y = 0.55;
ego.group.add(sensorGroup);
const beamGeo = new THREE.BoxGeometry(1, 0.035, 0.035).translate(0.5, 0, 0);
const hitGeo = new THREE.SphereGeometry(0.11, 12, 8);
const sensors: { beam: THREE.Mesh; hit: THREE.Mesh; material: THREE.MeshBasicMaterial }[] = [];
const dangerColor = new THREE.Color(PALETTE.collision);
function ensureSensors(count: number) {
  while (sensors.length < count) {
    const material = new THREE.MeshBasicMaterial({ color: PALETTE.sensor, transparent: true, opacity: 0.8, depthWrite: false, toneMapped: false });
    const beam = new THREE.Mesh(beamGeo, material);
    const hit = new THREE.Mesh(hitGeo, material);
    sensorGroup.add(beam, hit);
    sensors.push({ beam, hit, material });
  }
}

// ============================================================ cupo objetivo
const target = new THREE.Group();
const targetGlowMat = new THREE.MeshBasicMaterial({ color: PALETTE.success, transparent: true, opacity: 0.3, depthWrite: false });
const targetGlow = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), targetGlowMat);
targetGlow.rotation.x = -Math.PI / 2;
targetGlow.position.y = 0.016;
const targetMarkMat = new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, toneMapped: false });
const targetMark = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), targetMarkMat);
targetMark.rotation.x = -Math.PI / 2;
targetMark.position.y = 0.02;
target.add(targetGlow, targetMark);
scene.add(target);

function placeTarget(info: { x: number; y: number }, slot: Slot) {
  const w = slot.width - 0.35, d = slot.depth - 0.35;
  target.position.copy(toWorld(info.x, info.y));
  target.rotation.y = (slot.opening ?? (info.y > 0 ? -1 : 1)) < 0 ? 0 : Math.PI;   // la "P" se lee desde el pasillo
  targetGlow.scale.set(w, d, 1);
  targetMark.scale.set(w, d, 1);
  if (!targetMarkMat.map) targetMarkMat.map = targetTexture(d / w);
}

// ============================================================ obstáculos del episodio
const obstacleGroup = new THREE.Group();
scene.add(obstacleGroup);
const coneMat = std({ color: 0xff6a3d, roughness: 0.45, emissive: 0x3a1000 });
const coneBandMat = std({ color: 0xffffff, roughness: 0.3, emissive: 0x333333 });
const coneGeo = new THREE.ConeGeometry(0.2, 0.72, 24);
const coneBandGeo = new THREE.CylinderGeometry(0.115, 0.14, 0.12, 24);

function buildObstacles(layout: Layout) {
  obstacleGroup.clear();
  for (const o of layout.obstacles) {
    if (o.kind === "car") {
      // Los autos estacionados miran hacia el pasillo
      const car = makeCar({ color: PALETTE.parked[(o.color ?? 0) % PALETTE.parked.length] });
      car.group.position.copy(toWorld(o.x, o.y));
      const facing = o.facing ?? (o.y > 0 ? -1 : 1);   // hacia dónde abre su cupo
      car.group.rotation.y = o.angle + (facing > 0 ? Math.PI / 2 : -Math.PI / 2);
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
      addPillar(obstacleGroup, o.x, o.y, o.hx * 2);
    }
  }
}

// ============================================================ peatones (nivel 3)
const pedestrianGroup = new THREE.Group();
scene.add(pedestrianGroup);
const PERSON = {
  skin: [0xf1c7a6, 0xc68863, 0x8d5a3b].map((c) => std({ color: c, roughness: 0.7 })),
  shirts: [0xf472b6, 0x2dd4bf, 0xfbbf24, 0x60a5fa].map((c) => std({ color: c, roughness: 0.6 })),
  pants: std({ color: 0x3b3561, roughness: 0.8 }),
  hair: std({ color: 0x2a1d1a, roughness: 0.9 }),
  limb: new THREE.CapsuleGeometry(1, 1, 4, 10),
  head: new THREE.SphereGeometry(0.13, 16, 12),
  hairCap: new THREE.SphereGeometry(0.135, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2),
};
interface PedestrianMesh { group: THREE.Group; limbs: THREE.Group[]; phase: number; last: { x: number; y: number } | null }
const pedestrianMeshes: PedestrianMesh[] = [];

function capsule(material: THREE.Material, radius: number, length: number, y: number) {
  const mesh = shadowed(new THREE.Mesh(PERSON.limb, material), true, false);
  mesh.scale.set(radius, length / 2, radius);
  mesh.position.y = y;
  return mesh;
}

// Persona estilizada que mira hacia +x; piernas y brazos giran desde cadera y hombro
function makePedestrian(index: number): PedestrianMesh {
  const group = new THREE.Group();
  const shirt = PERSON.shirts[index % PERSON.shirts.length];
  const skin = PERSON.skin[index % PERSON.skin.length];
  group.add(capsule(shirt, 0.17, 0.4, 1.22));
  const head = shadowed(new THREE.Mesh(PERSON.head, skin), true, false);
  head.position.y = 1.66;
  const hair = new THREE.Mesh(PERSON.hairCap, PERSON.hair);
  hair.position.y = 1.68;
  group.add(head, hair);
  const limbs: THREE.Group[] = [];
  const parts: [number, number, THREE.Material, number, number][] = [
    [0.92, 0.09, PERSON.pants, 0.075, 0.72], [0.92, -0.09, PERSON.pants, 0.075, 0.72],
    [1.44, 0.24, skin, 0.05, 0.5], [1.44, -0.24, skin, 0.05, 0.5],
  ];
  for (const [y, z, material, radius, length] of parts) {
    const pivot = new THREE.Group();
    pivot.position.set(0, y, z);
    pivot.add(capsule(material, radius, length, -length / 2 - radius));
    group.add(pivot);
    limbs.push(pivot);
  }
  pedestrianGroup.add(group);
  return { group, limbs, phase: 0, last: null };
}

function ensurePedestrians(count: number) {
  while (pedestrianMeshes.length < count) pedestrianMeshes.push(makePedestrian(pedestrianMeshes.length));
  pedestrianMeshes.forEach((p, i) => {
    p.group.visible = i < count;
    p.last = null;
  });
}

function renderPedestrians() {
  const peds = sim.frame?.pedestrians ?? [];
  peds.forEach((info, i) => {
    const mesh = pedestrianMeshes[i];
    const pose = sim.pose.peds?.[i];
    if (!mesh || !pose) return;
    mesh.group.position.copy(toWorld(pose.x, pose.y));
    // Mira hacia donde camina (v6 trae su rumbo; antes solo subía o bajaba por el pasillo)
    mesh.group.rotation.y = info.heading !== undefined ? info.heading
      : (info.dir > 0 ? Math.PI / 2 : -Math.PI / 2);
    if (mesh.last) mesh.phase += Math.hypot(pose.x - mesh.last.x, pose.y - mesh.last.y) / 0.32;
    mesh.last = { x: pose.x, y: pose.y };
    // Balanceo de la marcha; al esperar, las extremidades vuelven al reposo
    const swing = info.moving ? Math.sin(mesh.phase) * 0.6 : 0;
    const [legL, legR, armL, armR] = mesh.limbs;
    legL.rotation.z = swing;
    legR.rotation.z = -swing;
    armL.rotation.z = -swing * 0.8;
    armR.rotation.z = swing * 0.8;
  });
}

// ============================================================ trayectoria
const TRAIL_MAX = 800;
const trailPoints: { x: number; y: number }[] = [];
const trailMesh = new THREE.Mesh(new THREE.BufferGeometry(),
  new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false, toneMapped: false }));
trailMesh.position.y = 0.024;
scene.add(trailMesh);

function rebuildTrail() {
  const n = trailPoints.length;
  if (n < 2) { trailMesh.geometry.setDrawRange(0, 0); return; }
  const positions = new Float32Array(n * 6), colors = new Float32Array(n * 6), indices: number[] = [];
  const color = new THREE.Color(), dim = new THREE.Color(THEME.trail[0]), bright = new THREE.Color(THEME.trail[1]);
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

function pushTrail(x: number, y: number) {
  const last = trailPoints[trailPoints.length - 1];
  if (last && Math.hypot(last.x - x, last.y - y) < 0.12) return;
  trailPoints.push({ x, y });
  if (trailPoints.length > TRAIL_MAX) trailPoints.shift();
  rebuildTrail();
}

// ============================================================ estado de la simulación
interface Pose {
  x: number; y: number; heading: number; steer: number; sensors: number[];
  peds: { x: number; y: number }[];
}

interface Sim {
  running: boolean; inFlight: boolean; lastStepAt: number; episodeOver: boolean; autoResetAt: number;
  speed: number; camera: CameraMode; showSensors: boolean; showTrail: boolean; level: LevelId;
  frame: Frame | null; carHalf: [number, number];
  from: Pose | null; to: Pose | null; animStart: number; animDuration: number;
  pose: Pose; statusAt: number;
  agent: string | null; held: number | null;
  queue: Frame[]; fetching: boolean; epoch: number; serverDone: boolean;
  garage: boolean; canTrain: boolean; catalog: CatalogItem[]; serving: Config["serving"];
}

const sim: Sim = {
  running: false, inFlight: false, lastStepAt: 0, episodeOver: false, autoResetAt: 0,
  speed: 1, camera: "orbit", showSensors: true, showTrail: true, level: "N2",
  frame: null, carHalf: [2.1, 0.9],
  from: null, to: null, animStart: 0, animDuration: BASE_STEP_MS,
  pose: { x: 0, y: 0, heading: 0, steer: 0, sensors: [], peds: [] },
  statusAt: 0,
  agent: null, held: null,   // held: acción que el usuario mantiene presionada en modo manual
  // Pasos del agente pedidos por adelantado: el servidor calcula varios por petición y aquí se
  // reproducen uno a uno, así la demora de internet no frena la animación.
  queue: [], fetching: false, epoch: 0, serverDone: false,
  garage: false, canTrain: false, catalog: [], serving: null,
};
const stepInterval = () => BASE_STEP_MS / sim.speed;
const batchSize = () => Math.min(64, Math.max(4, Math.ceil(1000 / stepInterval())));   // ~1 s de juego

function lerpAngle(a: number, b: number, t: number) {
  return a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * t;
}

function poseFromFrame(frame: Frame): Pose {
  return {
    x: frame.car.x, y: frame.car.y, heading: frame.car.heading, steer: frame.car.steer,
    sensors: frame.sensors.map((s) => s.distance),
    peds: frame.pedestrians.map((p) => ({ x: p.x, y: p.y })),
  };
}

function applyReset(frame: ResetFrame) {
  sim.garage = frame.layout.kind === "garage";
  if (!garageBuilt) (sim.garage ? buildRooftop : buildGarage)(frame.layout);
  sim.carHalf = [frame.car.length / 2, frame.car.width / 2];
  buildObstacles(frame.layout);
  const targetSlot = frame.layout.slots.find((s) => s.id === frame.target.id);
  if (!targetSlot) throw new Error(`El cupo objetivo ${frame.target.id} no está en el mapa`);
  placeTarget(frame.target, targetSlot);
  ensureSensors(frame.sensors.length);
  ensurePedestrians(frame.pedestrians.length);
  sim.pose = poseFromFrame(frame);
  sim.from = sim.to = sim.pose;
  sim.episodeOver = false;
  trailPoints.length = 0;
  pushTrail(frame.car.x, frame.car.y);
  $("banner").hidden = true;
  applyFrame(frame, false);
}

function applyFrame(frame: Frame, animate = true) {
  sim.frame = frame;
  if (animate) {
    sim.from = { ...sim.pose, sensors: [...sim.pose.sensors], peds: sim.pose.peds.map((p) => ({ ...p })) };
    sim.to = poseFromFrame(frame);
    sim.animStart = performance.now();
    sim.animDuration = stepInterval();
  }
  // Luces traseras: fuertes al frenar, medias en reversa
  ego.tailMat.emissiveIntensity = frame.action === BRAKE_ACTION ? 4 : frame.car.speed < -0.05 ? 2.5 : 1.2;
  if (["success", "collision", "timeout"].includes(frame.status) && !sim.episodeOver) {
    sim.episodeOver = true;
    sim.autoResetAt = performance.now() + AUTO_RESET_MS;
    sim.statusAt = performance.now();
    showBanner(frame.status, frame.collision_kind);
  }
  updateHud(frame);
  updateTraining(frame.training);
}

function updatePose(now: number) {
  if (!sim.to || !sim.from) return;
  const t = Math.min(1, (now - sim.animStart) / sim.animDuration);
  const { from, to } = sim;
  sim.pose = {
    x: from.x + (to.x - from.x) * t,
    y: from.y + (to.y - from.y) * t,
    heading: lerpAngle(from.heading, to.heading, t),
    steer: from.steer + (to.steer - from.steer) * t,
    sensors: to.sensors.map((d, i) => (from.sensors[i] ?? d) + (d - (from.sensors[i] ?? d)) * t),
    peds: to.peds.map((p, i) => {
      const f = from.peds[i] ?? p;
      return { x: f.x + (p.x - f.x) * t, y: f.y + (p.y - f.y) * t };
    }),
  };
}

// ============================================================ render por frame
const prevPos = new THREE.Vector2();

function renderCar(now: number) {
  const { pose } = sim;
  ego.group.position.copy(toWorld(pose.x, pose.y));
  ego.group.rotation.y = pose.heading;
  ego.steerPivots.forEach((p) => (p.rotation.y = pose.steer));

  const moved = Math.hypot(pose.x - prevPos.x, pose.y - prevPos.y);
  const speed = sim.frame?.car.speed ?? 0;
  ego.wheels.forEach((w) => (w.rotation.x -= Math.sign(speed) * moved / WHEEL_RADIUS));
  prevPos.set(pose.x, pose.y);
  if (sim.showTrail && moved > 0) pushTrail(pose.x, pose.y);

  arrowHolder.rotation.y = speed < -0.05 ? Math.PI : 0;
  arrowHolder.position.y = 1.9 + Math.sin(now / 300) * 0.06;

  // Indicador de colisión: anillo pulsante y carrocería en rojo
  const crashed = sim.frame?.status === "collision";
  const pulse = crashed ? 0.5 + 0.5 * Math.sin((now - sim.statusAt) / 90) : 0;
  collisionRing.material.opacity = crashed ? 0.4 + 0.5 * pulse : 0;
  collisionRing.scale.setScalar(crashed ? 1 + 0.15 * pulse : 1);
  ego.bodyMat.emissive.setHex(crashed ? PALETTE.collision : 0x000000);
  ego.bodyMat.emissiveIntensity = crashed ? 0.6 * pulse : 0;
  egoGlow.material.color.setHex(crashed ? PALETTE.collision : THEME.egoGlow);

  // Sensores
  const [hl, hw] = sim.carHalf;
  sensorGroup.visible = sim.showSensors;
  sim.frame?.sensors.forEach((s, i) => {
    const sensor = sensors[i];
    const range = s.range ?? sim.frame?.sensor_range ?? 8;
    const c = Math.cos(s.angle), sn = Math.sin(s.angle);
    const dist = Math.max(pose.sensors[i] ?? s.distance, 0.02);
    // Rayos radiales: salen del borde de la carrocería. Parachoques: desde su punto (adelante, izquierda)
    const body = Math.min(Math.abs(c) > 1e-6 ? hl / Math.abs(c) : Infinity, Math.abs(sn) > 1e-6 ? hw / Math.abs(sn) : Infinity);
    const [ox, oy] = s.origin ?? [c * body, sn * body];
    sensor.beam.position.set(ox, 0, -oy);
    sensor.beam.rotation.y = s.angle;
    sensor.beam.scale.x = dist;
    sensor.hit.position.set(ox + c * dist, 0, -(oy + sn * dist));
    sensor.hit.visible = dist < range - 0.05;
    const danger = THREE.MathUtils.clamp(1 - (dist - 0.5) / 2, 0, 1);
    sensor.material.color.setHex(PALETTE.sensor).lerp(dangerColor, danger);
    sensor.material.opacity = 0.4 + 0.5 * (1 - dist / range);
  });

  // Cupo objetivo: pulso suave, verde intenso al estacionar
  const success = sim.frame?.status === "success";
  targetGlowMat.opacity = success ? 0.65 + 0.1 * Math.sin(now / 120) : 0.28 + 0.1 * Math.sin(now / 420);
  targetMarkMat.color.setHex(0x4ade80).multiplyScalar(success ? 2.2 : 1.3 + 0.25 * Math.sin(now / 420));
}

const cameraTween = { active: false, start: 0, fromPos: new THREE.Vector3(), fromTarget: new THREE.Vector3(),
  toPos: new THREE.Vector3(), toTarget: new THREE.Vector3() };
type CameraMode = "orbit" | "top" | "follow";
const CAMERA_PRESETS: Record<Exclude<CameraMode, "follow">, { pos: Vec3; target: Vec3 }> = {
  orbit: { pos: [0, 10, 15.5], target: [0, 0, -0.5] },
  top: { pos: [0, 40, 0.01], target: [0, 0, 0] },
};

function setCamera(mode: CameraMode) {
  sim.camera = mode;
  document.querySelectorAll<HTMLElement>("#camera-select button").forEach((b) => b.classList.toggle("active", b.dataset.camera === mode));
  controls.enabled = false;
  if (mode === "follow") return;
  Object.assign(cameraTween, { active: true, start: performance.now() });
  cameraTween.fromPos.copy(camera.position);
  cameraTween.fromTarget.copy(controls.target);
  cameraTween.toPos.set(...CAMERA_PRESETS[mode].pos);
  cameraTween.toTarget.set(...CAMERA_PRESETS[mode].target);
}

function updateCamera(dt: number) {
  if (sim.camera === "follow") {
    const { x, y, heading } = sim.pose;
    const back = toWorld(x - Math.cos(heading) * 7, y - Math.sin(heading) * 7, 3.2);
    const look = toWorld(x + Math.cos(heading) * 3, y + Math.sin(heading) * 3, 0.7);
    const k = 1 - Math.exp(-dt * 3.5);
    camera.position.lerp(back, k);
    controls.target.lerp(look, k);
    camera.lookAt(controls.target);
  } else if (cameraTween.active) {
    const t = Math.min(1, (performance.now() - cameraTween.start) / 900);
    const e = 1 - Math.pow(1 - t, 3);
    camera.position.lerpVectors(cameraTween.fromPos, cameraTween.toPos, e);
    controls.target.lerpVectors(cameraTween.fromTarget, cameraTween.toTarget, e);
    camera.lookAt(controls.target);
    if (t >= 1) {
      cameraTween.active = false;
      controls.enabled = sim.camera === "orbit";
    }
  } else if (sim.camera === "orbit") {
    controls.update();
  }
  // El techo se oculta cuando la cámara sube por encima de él
  ceilingGroup.visible = camera.position.y < CEILING_H - 0.3;
}

function resize() {
  const { clientWidth: w, clientHeight: h } = viewport;
  renderer.setSize(w, h);
  composer.setPixelRatio(renderer.getPixelRatio());
  composer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);

let lastFrameTime = performance.now();
function tick(now: number) {
  const dt = Math.min(0.1, (now - lastFrameTime) / 1000);
  lastFrameTime = now;
  if (sim.running && !sim.inFlight) {
    if (sim.episodeOver && now >= sim.autoResetAt) doReset();
    else if (!sim.episodeOver && sim.agent !== "manual") {
      if (!sim.fetching && !sim.serverDone && sim.queue.length <= batchSize() / 2) fetchSteps();
      const next = sim.queue.length && now - sim.lastStepAt >= stepInterval() ? sim.queue.shift() : undefined;
      if (next) {
        applyFrame(next);
        sim.lastStepAt = now;
      }
    } else if (!sim.episodeOver && sim.held !== null && now - sim.lastStepAt >= stepInterval()) {
      // En modo manual el tiempo solo avanza mientras se mantiene presionada una acción
      doStep(sim.held);
    }
  }
  updatePose(now);
  renderCar(now);
  renderPedestrians();
  updateCamera(dt);
  composer.render();
  requestAnimationFrame(tick);
}

// ============================================================ HUD y panel
function loadingText(text: string) {
  const el = $("loading");
  if (el.lastChild) el.lastChild.textContent = text;
}

let noticeTimer: ReturnType<typeof setTimeout> | undefined;
const connection = new Connection((ok) => {
  if (!ok) loadingText("Sin conexión con el servidor. Reintentando…");
});

function showError(err: unknown) {
  const el = $("error");
  el.textContent = err instanceof Error ? err.message : String(err);
  el.hidden = false;
  $("panel").classList.add("open");
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => (el.hidden = true), 5000);
}

function showBanner(status: Frame["status"], collisionKind: Frame["collision_kind"]) {
  const el = $("banner");
  el.textContent = status === "collision"
    ? COLLISION_LABELS[collisionKind ?? ""] ?? "¡Choque!"
    : status === "success" ? "¡Estacionado correctamente!" : "Tiempo agotado";
  el.className = `banner ${status}`;
  el.hidden = false;
  const flash = $("flash");
  const FLASH: Record<string, string> = { success: "rgba(34,197,94,0.55)", collision: "rgba(239,68,68,0.65)", timeout: "rgba(245,158,11,0.45)" };
  flash.style.setProperty("--flash", FLASH[status] ?? FLASH.timeout);
  flash.classList.add("on");
  setTimeout(() => flash.classList.remove("on"), 700);
}

const setText = (id: string, text: string | number) => ($(id).textContent = String(text));
const degrees = (rad: number) => `${Math.round((rad * 180) / Math.PI)}°`;

function updateHud(frame: Frame) {
  let status: Status = frame.status;
  if (status === "running" && !sim.running) status = frame.steps === 0 ? "idle" : "paused";
  $("hud-title").dataset.state = status;
  setText("hud-title-text", sim.garage && ["idle", "running"].includes(status) ? "Encuentra el espacio libre" : TITLE_LABELS[status]);
  setText("hud-moves", `Movimientos: ${frame.steps}`);
  setText("hud-level", `Nivel: ${sim.level.replace("N", "")}`);

  setText("m-agent", frame.agent_label ?? frame.agent);
  setText("hud-mode", `Modo: ${frame.mode}`);
  setText("m-distance", `${frame.distance.toFixed(1)} m`);
  setText("m-speed", `${Math.round(Math.abs(frame.car.speed) * 3.6)} km/h`);
  const reward = $("m-reward");
  reward.textContent = frame.reward.toFixed(2);
  reward.className = frame.reward >= 0 ? "good" : "bad";
  setText("m-episode", frame.episode);
  setText("m-collisions", frame.collisions_total);

  setText("m-status", status === "collision" && frame.collision_kind
    ? `Choque · ${COLLISION_CAUSES[frame.collision_kind]}` : STATUS_LABELS[status]);
  setText("m-total", frame.total_reward.toFixed(1));
  setText("m-heading", degrees(frame.heading_error));
  setText("m-steer", degrees(frame.car.steer));
  setText("m-action", frame.action_name ?? "—");
  setText("m-successes", frame.successes_total);
  // Identidad del modelo activo: ambiente, commit, semilla y huella de los pesos
  const m = frame.model;
  setText("m-trained", m
    ? `Ambiente ${frame.environment_version} · commit ${m.git_commit?.slice(0, 7) ?? "?"} · semilla ${m.seed} · sha256 ${m.model_sha256?.slice(0, 10)}…`
    : `Ambiente ${frame.environment_version} · ${frame.mode}`);
  const link = $<HTMLAnchorElement>("m-mlflow");
  const exp = m?.mlflow_experiment ? MLFLOW_EXPERIMENTS[m.mlflow_experiment] : undefined;
  link.hidden = !(m?.mlflow_run_id && exp);
  if (m?.mlflow_run_id && exp) link.href = `${MLFLOW_URL}/#/experiments/${exp}/runs/${m.mlflow_run_id}`;
  $<HTMLSelectElement>("agent-select").value = frame.agent;
  $<HTMLButtonElement>("btn-train").disabled = frame.training.running || !sim.canTrain;
}

let wasTraining = false;
function updateTraining(training: Training) {
  if (training.running || training.total) {
    $("train-bar").style.width = `${(100 * training.episode) / Math.max(1, training.total)}%`;
    setText("train-label", training.running
      ? `${training.experiment_id} · ${training.episode}/${training.total}`
      : `${training.experiment_id} · ${training.error ? "falló" : "registrado " + (training.run_id ?? "")}`);
    setText("train-success", `éxito ${(training.success_rate * 100).toFixed(0)}%`);
  }
  if (training.error) showError(new Error(`Entrenamiento: ${training.error}`));
  drawSparkline(training.returns ?? []);
  if (wasTraining && !training.running) refreshConfig();
  wasTraining = training.running;
}

function drawSparkline(values: number[]) {
  const canvas = $<HTMLCanvasElement>("sparkline");
  const ratio = devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * ratio;
  canvas.height = h * ratio;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(ratio, ratio);
  if (values.length < 2) return;
  const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
  const y = (v: number) => h - 4 - ((v - min) / span) * (h - 8);
  if (min < 0 && max > 0) {
    ctx.strokeStyle = THEME.sparkAxis;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(0, y(0)); ctx.lineTo(w, y(0)); ctx.stroke();
    ctx.setLineDash([]);
  }
  const gradient = ctx.createLinearGradient(0, 0, w, 0);
  gradient.addColorStop(0, THEME.spark[0]);
  gradient.addColorStop(1, THEME.spark[1]);
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = (i / (values.length - 1)) * w;
    i ? ctx.lineTo(x, y(v)) : ctx.moveTo(x, y(v));
  });
  ctx.stroke();
}

async function refreshConfig() {
  const config = await connection.send("config");
  // Catálogo del nivel: referencias y modelos registrados (los fallidos aparecen deshabilitados)
  $<HTMLSelectElement>("agent-select").replaceChildren(...config.agents.map((a) => {
    const option = document.createElement("option");
    option.value = a.id;
    const rate = a.test_success_rate == null ? "" : ` · éxito ${Math.round(a.test_success_rate * 100)}%`;
    option.textContent = a.available ? `${a.label}${rate}`
      : a.status === "otra versión del nivel" ? `${a.label} · ${a.status}` : `${a.label} · ${a.status} (sin checkpoint)`;
    option.disabled = !a.available;
    return option;
  }));
  $<HTMLSelectElement>("agent-select").value = config.agent;
  setManual(config.agent === "manual");
  $<HTMLSelectElement>("difficulty").replaceChildren(...config.levels.map((l) => {
    const option = document.createElement("option");
    option.value = l.id;
    option.textContent = `${l.id} · ${l.name}`;
    return option;
  }));
  sim.level = config.level;
  sim.canTrain = config.can_train;
  $<HTMLSelectElement>("difficulty").value = config.level;
  $("train-section").hidden = !config.can_train;      // en la versión publicada se entrena en local
  sim.catalog = config.agents;
  sim.serving = config.serving;
  renderServing();
  renderEvalChoices();
  return config;
}

// ============================================================ modelo servido y rollback
const shortRun = (id: string | null | undefined) => id?.replace(/-[0-9a-f]{8}$/, "") ?? "—";

function renderServing() {
  const s = sim.serving;
  $<HTMLButtonElement>("btn-rollback").disabled = !s?.previous;
  $<HTMLButtonElement>("btn-use-served").disabled = !s;
  if (!s) {
    setText("srv-current", "Este nivel no tiene puntero de modelo servido: se usa el de mayor éxito en la suite.");
    $("srv-history").replaceChildren();
    return;
  }
  const c = s.current;
  const rate = c.test_success_rate == null ? "" : ` · éxito en la suite ${Math.round(c.test_success_rate * 100)}%`;
  setText("srv-current", `${shortRun(c.run_id)}${rate} · ${c.environment_version} · sha256 ${c.model_sha256.slice(0, 10)}…`);
  const ACTIONS = { promote: "Promovido", rollback: "Rollback" };
  $("srv-history").replaceChildren(...[...s.history].reverse().map((h) => {
    const li = document.createElement("li");
    const b = document.createElement("b");
    b.textContent = `${ACTIONS[h.action] ?? h.action} · ${shortRun(h.run_id)}`;
    li.append(b, ` — ${h.at.slice(0, 10)} · ${h.reason}`);
    return li;
  }));
}

async function useAgent(id: string, note?: string) {
  try {
    await connection.send("agent", { agent: id });
    $<HTMLSelectElement>("agent-select").value = id;
    setManual(false);
    await doReset();
    setText("eval-note", "");
    if (note) showNotice(note);
  } catch (err) { showError(err); }
}

$<HTMLButtonElement>("btn-use-served").onclick = () => sim.serving && useAgent(sim.serving.current.run_id,
  `Maneja el modelo servido: ${shortRun(sim.serving.current.run_id)}`);
$<HTMLButtonElement>("btn-rollback").onclick = () => sim.serving?.previous && useAgent(sim.serving.previous,
  `Rollback en esta partida: ahora maneja ${shortRun(sim.serving.previous)} (versión anterior). ` +
  "El rollback oficial queda en git con «python -m agents.serving rollback».");

function showNotice(text: string) {
  const el = $("error");
  el.textContent = text;
  el.classList.add("info");
  el.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { el.hidden = true; el.classList.remove("info"); }, 7000);
}

// ============================================================ evaluar y comparar
function renderEvalChoices() {
  const defaults = new Set([sim.serving?.current.run_id, sim.serving?.previous, "rule"].filter(Boolean));
  $("eval-agents").replaceChildren(...sim.catalog.filter((a) => a.available && a.id !== "manual").map((a) => {
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = a.id;
    box.checked = defaults.has(a.id);
    const rate = a.test_success_rate == null ? "" : ` · ${Math.round(a.test_success_rate * 100)}%`;
    label.append(box, `${a.label}${rate}`);
    return label;
  }));
}

const OUTCOME_SHORT: Record<string, string> = { success: "Éxito", obstacle: "Obstáculo", pedestrian: "Peatón", boundary: "Salida", timeout: "Tiempo" };
const pct = (x: number) => `${Math.round(x * 100)}%`;

$<HTMLButtonElement>("btn-eval").onclick = async () => {
  const agents = [...$("eval-agents").querySelectorAll<HTMLInputElement>("input:checked")].map((i) => i.value);
  const episodes = Number($<HTMLInputElement>("eval-episodes").value);
  if (!agents.length || agents.length > 4) return showError(new Error("Elige de 1 a 4 agentes para comparar"));
  if (!Number.isInteger(episodes) || episodes < 1 || episodes > 30) return showError(new Error("Los casos deben ser de 1 a 30"));
  $<HTMLButtonElement>("btn-eval").disabled = true;
  setText("eval-note", "Evaluando en el servidor…");
  try {
    const res = await connection.send("evaluate", { agents, episodes });
    const best = Math.max(...res.results.map((r) => r.summary.success_rate));
    const head = document.createElement("tr");
    head.innerHTML = "<th>Agente</th><th>Éxito</th><th>Tiempo</th><th>Fallos</th>";
    $("eval-table").replaceChildren(head, ...res.results.map((r) => {
      const tr = document.createElement("tr");
      tr.className = r.summary.success_rate === best ? "best" : "";
      const fails = Object.entries(r.summary.outcomes).filter(([o, n]) => o !== "success" && n)
        .map(([o, n]) => `${OUTCOME_SHORT[o]} ${n}`).join(", ") || "—";
      for (const [text, cls] of [[shortRun(r.label.replace(/^[A-Z]+ · /, "")), ""], [pct(r.summary.success_rate), "num"],
                                 [`${r.summary.mean_time_s.toFixed(1)} s`, "num"], [fails, ""]]) {
        const td = document.createElement("td");
        td.textContent = text;
        td.className = cls;
        tr.append(td);
      }
      tr.title = r.model_sha256 ? `${r.agent}\nsha256 ${r.model_sha256}\ncommit ${r.git_commit}` : r.agent;
      return tr;
    }));
    $("eval-table").hidden = false;
    setText("eval-note", `${res.suite_id} · semillas ${res.seed_start}–${res.seed_start + res.episodes - 1} · ${res.level}`);
  } catch (err) {
    showError(err);
    setText("eval-note", "");
  } finally {
    $<HTMLButtonElement>("btn-eval").disabled = false;
  }
};

// ============================================================ monitoreo
async function refreshMetrics() {
  if (!$("panel").classList.contains("open")) return;
  try {
    const m = await fetch(`${API_BASE}/api/metrics`).then((r) => {
      if (!r.ok) throw new Error(`Monitoreo: el servidor respondió ${r.status}`);
      return r.json();
    });
    setText("mon-sessions", m.live_sessions);
    setText("mon-latency", `${m.step_latency_ms.p95.toFixed(1)} ms`);
    setText("mon-commit", m.git_commit?.slice(0, 7) ?? "local");
    const head = document.createElement("tr");
    head.innerHTML = "<th>Nivel</th><th>Episodios</th><th>Éxito</th><th>Fallos</th><th>Servido</th>";
    const rows = Object.keys(m.served).map((lid) => {
      const lv = m.levels[lid];
      const tr = document.createElement("tr");
      const fails = lv ? Object.entries(lv.outcomes).filter(([o, n]) => o !== "success" && n)
        .map(([o, n]) => `${OUTCOME_SHORT[o]} ${n}`).join(", ") || "—" : "—";
      for (const [text, cls] of [[lid, ""], [lv?.episodes ?? 0, "num"], [lv ? pct(lv.success_rate) : "—", "num"],
                                 [fails, ""], [shortRun(m.served[lid]), ""]]) {
        const td = document.createElement("td");
        td.textContent = text;
        td.className = cls;
        tr.append(td);
      }
      return tr;
    });
    $("mon-table").replaceChildren(head, ...rows);
    setText("mon-note", `Desde que arrancó el servidor (${Math.round(m.uptime_s / 60)} min) · ${m.steps} pasos · latencia media ${m.step_latency_ms.mean.toFixed(2)} ms por paso`);
  } catch (err) {
    setText("mon-note", `Sin datos de monitoreo: ${err instanceof Error ? err.message : err}`);
  }
}
setInterval(refreshMetrics, 5000);
$("btn-panel").addEventListener("click", () => setTimeout(refreshMetrics, 50));

async function doStep(action?: number) {
  sim.inFlight = true;
  try {
    applyFrame(await connection.send("step", action === undefined ? {} : { action }));
  } catch (err) {
    showError(err);
    setRunning(false);
  } finally {
    sim.lastStepAt = performance.now();
    sim.inFlight = false;
  }
}

async function fetchSteps() {
  const epoch = sim.epoch;
  sim.fetching = true;
  try {
    const { frames } = await connection.send("steps", { n: batchSize() });
    if (epoch !== sim.epoch) return;              // llegó tarde: ya se reinició o se cambió de agente
    sim.queue.push(...frames);
    sim.serverDone = frames[frames.length - 1].status !== "running";
  } catch (err) {
    if (epoch === sim.epoch) { showError(err); setRunning(false); }
  } finally {
    sim.fetching = false;
  }
}

function dropQueue() {
  sim.epoch += 1;
  sim.queue = [];
  sim.serverDone = false;
}

async function doReset(payload = {}) {
  dropQueue();
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

function setRunning(running: boolean) {
  sim.running = running;
  $("btn-play").classList.toggle("playing", running);
  if (sim.frame) updateHud(sim.frame);
}

function setOverlays(sensorsOn: boolean, trailOn: boolean) {
  sim.showSensors = sensorsOn;
  sim.showTrail = trailOn;
  trailMesh.visible = trailOn;
  $<HTMLInputElement>("tg-sensors").checked = sensorsOn;
  $<HTMLInputElement>("tg-trail").checked = trailOn;
  $("btn-hint").classList.toggle("on", sensorsOn || trailOn);
}

$("btn-play").onclick = () => {
  if (sim.running) return setRunning(false);
  if (sim.episodeOver) doReset();
  setRunning(true);
};
$("btn-reset").onclick = () => doReset();
$("btn-hint").onclick = () => {
  const on = !(sim.showSensors || sim.showTrail);
  setOverlays(on, on);
};
const NEXT_CAMERA: Record<CameraMode, CameraMode> = { orbit: "top", top: "follow", follow: "orbit" };
$("btn-camera").onclick = () => setCamera(NEXT_CAMERA[sim.camera]);
$("btn-panel").onclick = () => $("panel").classList.toggle("open");
$("btn-panel-close").onclick = () => $("panel").classList.remove("open");

$<HTMLSelectElement>("agent-select").onchange = async (e) => {
  const value = (e.target as HTMLSelectElement).value;
  try {
    await connection.send("agent", { agent: value });
    setManual(value === "manual");
    await doReset();
  } catch (err) {
    showError(err);
    await refreshConfig();
  }
};
document.querySelectorAll<HTMLElement>("#camera-select button").forEach((b) => (b.onclick = () => setCamera(b.dataset.camera as CameraMode)));

$<HTMLInputElement>("speed").oninput = (e) => {
  sim.speed = Number((e.target as HTMLInputElement).value);
  setText("speed-label", `${sim.speed}×`);
};
$<HTMLSelectElement>("difficulty").onchange = async (e) => {
  const level: LevelId = (e.target as HTMLSelectElement).value;
  sim.level = currentLevel = level;
  const url = new URL(location.href);
  url.searchParams.set("level", currentLevel);
  history.replaceState(null, "", url);
  await doReset({ level });
  if (THEMES_DIFFER(level)) return location.reload();
  await refreshConfig();
};
$<HTMLInputElement>("tg-sensors").onchange = (e) => setOverlays((e.target as HTMLInputElement).checked, sim.showTrail);
$<HTMLInputElement>("tg-trail").onchange = (e) => setOverlays(sim.showSensors, (e.target as HTMLInputElement).checked);
$<HTMLInputElement>("tg-noise").onchange = (e) => doReset({ sensor_noise: (e.target as HTMLInputElement).checked ? 0.1 : 0 });

$<HTMLButtonElement>("btn-train").onclick = async () => {
  const episodes = Number($<HTMLInputElement>("train-episodes").value);
  if (!Number.isInteger(episodes) || episodes < 1 || episodes > 50000) {
    showError(new Error("Los episodios deben ser un entero entre 1 y 50000"));
    return;
  }
  try {
    const res = await connection.send("train", { episodes });
    updateTraining(res.training);
    $<HTMLButtonElement>("btn-train").disabled = true;
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

// ============================================================ control manual
const MANUAL_KEYS = Object.fromEntries(
  [...document.querySelectorAll<HTMLElement>("#dpad [data-action]")].map((b) => [b.dataset.key, Number(b.dataset.action)]));

function setManual(on: boolean) {
  sim.agent = on ? "manual" : $<HTMLSelectElement>("agent-select").value;
  $("dpad").hidden = !on;
  hold(null);
}

function hold(action: number | null) {
  sim.held = action;
  document.querySelectorAll<HTMLElement>("#dpad [data-action]").forEach((b) => b.classList.toggle("held", Number(b.dataset.action) === action));
  if (action !== null && !sim.running) $("btn-play").click();   // al primer movimiento arranca el episodio
}

document.querySelectorAll<HTMLElement>("#dpad [data-action]").forEach((b) => {
  b.onpointerdown = (e) => { e.preventDefault(); b.setPointerCapture(e.pointerId); hold(Number(b.dataset.action)); };
  b.onpointerup = b.onpointercancel = () => hold(null);
});
window.addEventListener("keyup", (e) => {
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (sim.agent === "manual" && MANUAL_KEYS[key] === sim.held) hold(null);
});
window.addEventListener("blur", () => hold(null));

window.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName ?? "")) return;
  const manualKey = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (sim.agent === "manual" && manualKey in MANUAL_KEYS) {
    e.preventDefault();
    if (!e.repeat) hold(MANUAL_KEYS[manualKey]);
    return;
  }
  const key = e.key.toLowerCase();
  if (key === " ") { e.preventDefault(); $("btn-play").click(); }
  else if (key === "r") doReset();
  else if (key === "c") $("btn-camera").click();
  else if (key === "s") setOverlays(!sim.showSensors, sim.showTrail);
  else if (key === "t") setOverlays(sim.showSensors, !sim.showTrail);
  else if (key === "p") $("btn-panel").click();
});

// ============================================================ inicio
(async () => {
  await document.fonts.load("600 40px Fredoka").catch(() => {});   // las texturas usan la fuente
  resize();
  setRunning(false);
  setOverlays(true, true);
  setCamera("orbit");
  try {
    loadingText("Cargando los autos…");
    await loadCarModel();
    mountEgoCar();
    requestAnimationFrame(tick);
    loadingText("Conectando con el simulador…");
    await refreshConfig();
    await doReset();
    $("loading").hidden = true;
    clearTimeout(wakeNotice);
    if (sim.agent !== "manual") setRunning(true);   // arranca solo: el auto empieza a andar al abrir
  } catch (err) {
    showError(err);
  }
})();
