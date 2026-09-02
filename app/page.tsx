"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { MarchingCubes } from "three/examples/jsm/objects/MarchingCubes.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";

type Method = "potential" | "velocity" | "vorticity" | "fd_laplacian";
type Sphere = [number, number, number, number];
type BoundaryMode = "mask" | "wire" | "hidden";
type Inference = {
  request_id: number;
  request_key: string;
  method: Method;
  mode: number;
  component: string;
  grid: number;
  scale: number;
  elapsed_ms: number;
  fluid_fraction: number;
  cloud: Float32Array;
  volume: Float32Array;
  slices: Record<"x" | "y" | "z", Float32Array>;
  vector_slices: Record<"x" | "y" | "z", Float32Array>;
  contract: string;
};

type Timings = { inference: number | null; network: number | null; parse: number | null; isosurface: number | null; render: number | null };
type IsoGeometry = { positions: Float32Array; normals: Float32Array };
type ObstacleGeometry = IsoGeometry & { key: string };
type BinaryArray = { offset: number; length: number; shape: number[] };
type BinaryHeader = Omit<Inference, "request_id" | "cloud" | "volume" | "slices" | "vector_slices"> & { format: string; arrays: Record<string, BinaryArray> };

const EMPTY_TIMINGS: Timings = { inference: null, network: null, parse: null, isosurface: null, render: null };
const LOCAL_BUILD = process.env.NEXT_PUBLIC_LOCAL_BUILD === "1";

function parseInferenceBinary(buffer: ArrayBuffer, requestId: number, requestKey: string): Inference {
  const headerLength = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength))) as BinaryHeader;
  if (header.format !== "eigenfluid-fp32-v1") throw new Error("Unsupported inference payload");
  const payloadOffset = (4 + headerLength + 3) & ~3;
  const read = (name: string) => {
    const descriptor = header.arrays[name];
    if (!descriptor) throw new Error(`Missing binary field: ${name}`);
    return new Float32Array(buffer, payloadOffset + descriptor.offset, descriptor.length);
  };
  return {
    request_id: requestId, request_key: requestKey, method: header.method, mode: header.mode, component: header.component,
    grid: header.grid, scale: header.scale, elapsed_ms: header.elapsed_ms,
    fluid_fraction: header.fluid_fraction, contract: header.contract,
    cloud: read("cloud"), volume: read("volume"),
    slices: { x: read("slices.x"), y: read("slices.y"), z: read("slices.z") },
    vector_slices: { x: read("vector_slices.x"), y: read("vector_slices.y"), z: read("vector_slices.z") },
  };
}

const METHODS: { id: Method; short: string; name: string; meta: string; metric: string }[] = [
  { id: "potential", short: "P", name: "Potential", meta: "∫|∇A|² / ∫|A|²", metric: "λ̄val 1,462.26" },
  { id: "velocity", short: "U", name: "Velocity", meta: "∫|∇u|² / ∫|u|²", metric: "λ̄val 2,919.97" },
  { id: "vorticity", short: "Ω", name: "Vorticity", meta: "∫|∇ω|² / ∫|ω|²", metric: "λ̄val 18,837.02" },
];

const INITIAL: Sphere[] = [
  [0.5000, 0.6800, 0.5000, 0.17],
  [0.3441, 0.4100, 0.5000, 0.17],
  [0.6559, 0.4100, 0.5000, 0.17],
];

function tone(value: number) {
  const v = Math.max(-1, Math.min(1, value));
  if (v >= 0) {
    const l = 92 - 51 * v;
    return `hsl(7 72% ${l}%)`;
  }
  return `hsl(211 78% ${92 - 48 * -v}%)`;
}

function Slice({ axis, values, spheres }: { axis: "x" | "y" | "z"; values?: Float32Array; spheres: Sphere[] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !values?.length) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const n = Math.round(Math.sqrt(values.length));
    canvas.width = n;
    canvas.height = n;
    const image = ctx.createImageData(n, n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const v = Math.max(-1, Math.min(1, values[(n - 1 - y) * n + x]));
      const t = Math.abs(v);
      const cold = [36, 104, 171], hot = [197, 51, 62], paper = [244, 241, 235];
      const target = v >= 0 ? hot : cold;
      const i = 4 * (y * n + x);
      image.data[i] = paper[0] + (target[0] - paper[0]) * t;
      image.data[i + 1] = paper[1] + (target[1] - paper[1]) * t;
      image.data[i + 2] = paper[2] + (target[2] - paper[2]) * t;
      image.data[i + 3] = 255;
    }
    const outer = new Float32Array(n * n);
    const obstacle = new Float32Array(n * n);
    const point = (row: number, column: number): [number, number, number] => {
      const a = row / (n - 1), b = column / (n - 1);
      if (axis === "x") return [0.5, a, b];
      if (axis === "y") return [a, 0.5, b];
      return [a, b, 0.5];
    };
    for (let row = 0; row < n; row++) for (let column = 0; column < n; column++) {
      const p = point(row, column);
      outer[row * n + column] = 1 - p.reduce((sum, value) => sum + ((value - 0.5) / 0.49) ** 8, 0);
      const distances = spheres.map(sphere => Math.hypot(p[0] - sphere[0], p[1] - sphere[1], p[2] - sphere[2]) - sphere[3]);
      const minimum = Math.min(...distances);
      obstacle[row * n + column] = minimum - 0.02 * Math.log(distances.reduce((sum, distance) => sum + Math.exp(-(distance - minimum) / 0.02), 0));
    }
    const crosses = (field: Float32Array, row: number, column: number) => {
      const center = field[row * n + column] >= 0;
      return (row > 0 && (field[(row - 1) * n + column] >= 0) !== center) ||
        (row + 1 < n && (field[(row + 1) * n + column] >= 0) !== center) ||
        (column > 0 && (field[row * n + column - 1] >= 0) !== center) ||
        (column + 1 < n && (field[row * n + column + 1] >= 0) !== center);
    };
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const row = n - 1 - y;
      const i = 4 * (y * n + x);
      const outerBoundary = crosses(outer, row, x);
      const obstacleBoundary = crosses(obstacle, row, x) && outer[row * n + x] >= 0;
      if (outerBoundary) { image.data[i] = 21; image.data[i + 1] = 32; image.data[i + 2] = 36; }
      if (obstacleBoundary) { image.data[i] = 36; image.data[i + 1] = 104; image.data[i + 2] = 162; }
    }
    ctx.putImageData(image, 0, 0);
  }, [axis, values, spheres]);
  return <figure className="slice"><canvas ref={ref} /><figcaption>{axis} = 0.5</figcaption></figure>;
}

function buildIsoSurface(values: number[], resolution: number, color: number, isolation: number, select: (value: number) => number, opacity = 0.72) {
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.38, metalness: 0.04, transparent: opacity < 1, opacity, side: THREE.DoubleSide, depthWrite: opacity > 0.8 });
  const surface = new MarchingCubes(resolution, material, false, false, 240000);
  surface.isolation = isolation;
  for (let x = 0; x < resolution; x++) for (let y = 0; y < resolution; y++) for (let z = 0; z < resolution; z++) {
    const index = (x * resolution + y) * resolution + z;
    surface.setCell(x, y, z, select(values[index] ?? 0));
  }
  surface.scale.setScalar(0.5);
  surface.update();
  return surface;
}

function superellipsePath() {
  const points: string[] = [];
  for (let index = 0; index <= 256; index++) {
    const angle = 2 * Math.PI * index / 256;
    const c = Math.cos(angle), s = Math.sin(angle);
    const radius = 0.49 / (Math.abs(c) ** 8 + Math.abs(s) ** 8) ** (1 / 8);
    points.push(`${index ? "L" : "M"}${(0.5 + radius * c).toFixed(4)},${(0.5 - radius * s).toFixed(4)}`);
  }
  return `${points.join(" ")} Z`;
}

function obstacleContourPath(axis: "x" | "y" | "z", spheres: Sphere[]) {
  const resolution = 128;
  const field = new Float32Array(resolution * resolution);
  const level = (a: number, b: number) => {
    const p = axis === "x" ? [0.5, a, b] : axis === "y" ? [a, 0.5, b] : [a, b, 0.5];
    const distances = spheres.map(sphere => Math.hypot(p[0] - sphere[0], p[1] - sphere[1], p[2] - sphere[2]) - sphere[3]);
    const minimum = Math.min(...distances);
    return minimum - 0.02 * Math.log(distances.reduce((sum, distance) => sum + Math.exp(-(distance - minimum) / 0.02), 0));
  };
  for (let row = 0; row < resolution; row++) for (let column = 0; column < resolution; column++) field[row * resolution + column] = level(row / (resolution - 1), column / (resolution - 1));
  const path: string[] = [];
  const interpolate = (a: [number, number], b: [number, number], va: number, vb: number): [number, number] => {
    const t = va === vb ? 0.5 : va / (va - vb);
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  };
  for (let row = 0; row < resolution - 1; row++) for (let column = 0; column < resolution - 1; column++) {
    const p: [number, number][] = [[column, row], [column + 1, row], [column + 1, row + 1], [column, row + 1]];
    const v = [field[row * resolution + column], field[row * resolution + column + 1], field[(row + 1) * resolution + column + 1], field[(row + 1) * resolution + column]];
    const hits: [number, number][] = [];
    for (let edge = 0; edge < 4; edge++) {
      const next = (edge + 1) % 4;
      if ((v[edge] >= 0) !== (v[next] >= 0)) hits.push(interpolate(p[edge], p[next], v[edge], v[next]));
    }
    for (let hit = 0; hit + 1 < hits.length; hit += 2) {
      const a = hits[hit], b = hits[hit + 1];
      path.push(`M${(a[0] / (resolution - 1)).toFixed(4)},${(1 - a[1] / (resolution - 1)).toFixed(4)}L${(b[0] / (resolution - 1)).toFixed(4)},${(1 - b[1] / (resolution - 1)).toFixed(4)}`);
    }
  }
  return path.join(" ");
}

function VectorSlice({ axis, vectors, spheres }: { axis: "x" | "y" | "z"; vectors?: Float32Array; spheres: Sphere[] }) {
  const boundary = useMemo(() => obstacleContourPath(axis, spheres), [axis, spheres]);
  const arrows = useMemo(() => {
    if (!vectors?.length) return [];
    const n = Math.round(Math.sqrt(vectors.length / 3)), step = Math.max(1, Math.floor(n / 13));
    const result: { x1: number; y1: number; x2: number; y2: number; opacity: number }[] = [];
    for (let row = Math.floor(step / 2); row < n; row += step) for (let column = Math.floor(step / 2); column < n; column += step) {
      const base = (row * n + column) * 3;
      const value = [vectors[base], vectors[base + 1], vectors[base + 2]];
      const projected = axis === "x" ? [value[2], -value[1]] : axis === "y" ? [value[2], -value[0]] : [value[1], -value[0]];
      const magnitude = Math.hypot(projected[0], projected[1]);
      if (magnitude < 0.035) continue;
      const length = 0.018 + 0.036 * Math.min(1, magnitude);
      const dx = projected[0] / magnitude * length, dy = projected[1] / magnitude * length;
      const x = column / (n - 1), y = 1 - row / (n - 1);
      result.push({ x1:x - dx / 2, y1:y - dy / 2, x2:x + dx / 2, y2:y + dy / 2, opacity:0.35 + 0.65 * Math.min(1, magnitude) });
    }
    return result;
  }, [axis, vectors]);
  const markerId = `arrow-${axis}`;
  return <figure className="slice vector-slice"><svg viewBox="0 0 1 1" role="img" aria-label={`${axis} center slice vector field`}>
    <defs><marker id={markerId} viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#bd3b48" /></marker></defs>
    <rect width="1" height="1" fill="#fbfaf6" />
    <g transform="translate(0.04 0.04) scale(0.92)">
      <path d={superellipsePath()} fill="#ececea" />
      {arrows.map((arrow, index) => <line key={index} {...arrow} stroke="#bd3b48" strokeWidth="0.0045" markerEnd={`url(#${markerId})`} />)}
      <path d={superellipsePath()} fill="none" stroke="#aeb4b2" strokeWidth="0.006" />
      <path d={boundary} fill="none" stroke="#2468a2" strokeWidth="0.009" strokeLinecap="round" strokeLinejoin="round" />
    </g>
  </svg><figcaption>{axis} = 0.5</figcaption></figure>;
}

function buildMetaballObstacle(spheres: Sphere[]) {
  const resolution = 42;
  const values = new Array<number>(resolution ** 3);
  for (let x = 0; x < resolution; x++) for (let y = 0; y < resolution; y++) for (let z = 0; z < resolution; z++) {
    const px = x / (resolution - 1), py = y / (resolution - 1), pz = z / (resolution - 1);
    let field = 0;
    for (const [sx, sy, sz, radius] of spheres) field += Math.exp((radius - Math.hypot(px - sx, py - sy, pz - sz)) / 0.032);
    values[(x * resolution + y) * resolution + z] = field;
  }
  return buildIsoSurface(values, resolution, 0x1e5f8a, 1, value => value, 1);
}

function buildIsoMesh(data: IsoGeometry, color: number, opacity: number) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(data.positions, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(data.normals, 3));
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.38, metalness: 0.04, transparent: opacity < 1, opacity, side: THREE.DoubleSide, depthWrite: opacity > 0.8 });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.scale.setScalar(0.5);
  return mesh;
}

function buildOuterBoundary(mode: Exclude<BoundaryMode, "hidden">) {
  const geometry = new RoundedBoxGeometry(1, 1, 1, 10, 0.16);
  if (mode === "wire") return new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry),
    new THREE.LineBasicMaterial({ color:0x8b9291, transparent:true, opacity:0.48 }),
  );
  return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color:0xd7dad8, transparent:true, opacity:0.18, side:THREE.DoubleSide, depthWrite:false }));
}

function FieldScene({ data, spheres, solidObstacle, surfaceModes, boundaryMode, onTiming }: { data: Inference | null; spheres: Sphere[]; solidObstacle: boolean; surfaceModes: boolean; boundaryMode: BoundaryMode; onTiming: (timing: Partial<Timings>) => void }) {
  const mount = useRef<HTMLDivElement>(null);
  const liveSphereMeshes = useRef<THREE.Mesh[]>([]);
  const [modeIso, setModeIso] = useState<IsoGeometry | null>(null);
  const [obstacleIso, setObstacleIso] = useState<ObstacleGeometry | null>(null);
  const sphereKey = useMemo(() => JSON.stringify(spheres), [spheres]);

  useEffect(() => {
    if (!surfaceModes || !data?.volume.length) {
      setModeIso(null);
      onTiming({ isosurface: 0 });
      return;
    }
    setModeIso(null);
    onTiming({ isosurface: null, render: null });
    const worker = new Worker(new URL("./isosurface.worker.ts", import.meta.url), { type: "module" });
    const modeBuffer = data.volume.slice().buffer as ArrayBuffer;
    worker.onmessage = (event: MessageEvent<{ mode: IsoGeometry | null; elapsedMs: number }>) => {
      setModeIso(event.data.mode);
      onTiming({ isosurface: event.data.elapsedMs });
    };
    worker.postMessage({ mode: { values: modeBuffer, resolution: data.grid }, obstacle: null }, [modeBuffer]);
    return () => worker.terminate();
  }, [data, surfaceModes, onTiming]);

  useEffect(() => {
    if (!solidObstacle) {
      setObstacleIso(null);
      return;
    }
    const worker = new Worker(new URL("./isosurface.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (event: MessageEvent<{ obstacle: IsoGeometry | null }>) => {
      if (event.data.obstacle) setObstacleIso({ ...event.data.obstacle, key: sphereKey });
    };
    worker.postMessage({ mode: null, obstacle: { spheres, resolution: 42 } });
    return () => worker.terminate();
  }, [spheres, sphereKey, solidObstacle]);

  useEffect(() => {
    const host = mount.current;
    if (!host) return;
    const renderStarted = performance.now();
    const width = host.clientWidth, height = host.clientHeight;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#edeae2");
    scene.fog = new THREE.Fog("#edeae2", 2.1, 3.4);
    const camera = new THREE.PerspectiveCamera(34, width / height, 0.01, 10);
    camera.position.set(1.28, 1.05, 1.55);
    camera.lookAt(0, 0, 0);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);
    scene.add(new THREE.HemisphereLight(0xf8f4ea, 0x7790a0, 2.25));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
    keyLight.position.set(1.4, 1.8, 1.2);
    scene.add(keyLight);
    if (boundaryMode !== "hidden") group.add(buildOuterBoundary(boundaryMode));
    const proxyMeshes = spheres.map((s, i) => {
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(1, 28, 18),
        new THREE.MeshBasicMaterial({ color: i === 1 ? 0xb73d49 : 0x2e6fa5, transparent: true, opacity: solidObstacle ? 0.2 : 0.12, wireframe: !solidObstacle, depthWrite: false }),
      );
      mesh.position.set(s[0] - 0.5, s[1] - 0.5, s[2] - 0.5);
      mesh.scale.setScalar(s[3]);
      mesh.visible = !solidObstacle || obstacleIso?.key !== sphereKey;
      group.add(mesh);
      return mesh;
    });
    liveSphereMeshes.current = proxyMeshes;
    if (solidObstacle && obstacleIso) group.add(buildIsoMesh(obstacleIso, 0x1e5f8a, 1));
    if (surfaceModes && modeIso) group.add(buildIsoMesh(modeIso, 0xbd3b48, 0.78));
    if (!surfaceModes && data?.cloud?.length) {
      const count = data.cloud.length / 4;
      const positions = new Float32Array(count * 3);
      const colors = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) {
        positions[3 * i] = data.cloud[4 * i] - 0.5; positions[3 * i + 1] = data.cloud[4 * i + 1] - 0.5; positions[3 * i + 2] = data.cloud[4 * i + 2] - 0.5;
        const color = new THREE.Color(tone(data.cloud[4 * i + 3]));
        colors[3 * i] = color.r; colors[3 * i + 1] = color.g; colors[3 * i + 2] = color.b;
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      const points = new THREE.Points(geometry, new THREE.PointsMaterial({ size: 0.018, vertexColors: true, transparent: true, opacity: 0.82, sizeAttenuation: true }));
      group.add(points);
    }
    let dragging = false, previousX = 0, previousY = 0;
    const down = (e: PointerEvent) => { dragging = true; previousX = e.clientX; previousY = e.clientY; renderer.domElement.setPointerCapture(e.pointerId); };
    const move = (e: PointerEvent) => { if (!dragging) return; group.rotation.y += (e.clientX - previousX) * 0.008; group.rotation.x += (e.clientY - previousY) * 0.006; previousX = e.clientX; previousY = e.clientY; };
    const up = () => { dragging = false; };
    renderer.domElement.addEventListener("pointerdown", down); renderer.domElement.addEventListener("pointermove", move); renderer.domElement.addEventListener("pointerup", up);
    renderer.render(scene, camera);
    const timingFrame = requestAnimationFrame(() => onTiming({ render: performance.now() - renderStarted }));
    let frame = 0;
    const draw = () => { frame = requestAnimationFrame(draw); renderer.render(scene, camera); };
    draw();
    const resize = () => { const w = host.clientWidth, h = host.clientHeight; camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h); };
    const observer = new ResizeObserver(resize); observer.observe(host);
    return () => { liveSphereMeshes.current = []; cancelAnimationFrame(frame); cancelAnimationFrame(timingFrame); observer.disconnect(); renderer.dispose(); host.removeChild(renderer.domElement); };
  }, [data, solidObstacle, surfaceModes, boundaryMode, modeIso, obstacleIso, onTiming]);

  useEffect(() => {
    liveSphereMeshes.current.forEach((mesh, index) => {
      const sphere = spheres[index];
      if (!sphere) return;
      mesh.position.set(sphere[0] - 0.5, sphere[1] - 0.5, sphere[2] - 0.5);
      mesh.scale.setScalar(sphere[3]);
      mesh.visible = !solidObstacle || obstacleIso?.key !== sphereKey;
    });
  }, [spheres, sphereKey, solidObstacle, obstacleIso]);
  return <div className="scene" ref={mount}><span className="scene-hint">drag to orbit</span></div>;
}

function Range({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return <label className="range"><span>{label}<b>{value.toFixed(2)}</b></span><input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))} /></label>;
}

export default function Home() {
  const [method, setMethod] = useState<Method>("potential");
  const [mode, setMode] = useState(0);
  const [component, setComponent] = useState("magnitude");
  const [spheres, setSpheres] = useState<Sphere[]>(INITIAL);
  const [data, setData] = useState<Inference | null>(null);
  const [eigenvalue, setEigenvalue] = useState<number | null>(null);
  const [eigenBusy, setEigenBusy] = useState(false);
  const [status, setStatus] = useState("Loading exact checkpoint…");
  const [timings, setTimings] = useState<Timings>(EMPTY_TIMINGS);
  const [busy, setBusy] = useState(false);
  const requestSequence = useRef(0);
  const [solidObstacle, setSolidObstacle] = useState(true);
  const [surfaceModes, setSurfaceModes] = useState(true);
  const [boundaryMode, setBoundaryMode] = useState<BoundaryMode>("mask");
  const [serverOnline, setServerOnline] = useState<boolean | null>(null);
  const active = METHODS.find(item => item.id === method)!;
  const liveEigenvalue = eigenBusy ? "λ computing…" : eigenvalue === null ? "λ pending" : `λ ${eigenvalue.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  const updateSphere = useCallback((index: number, field: number, value: number) => {
    setSpheres(current => current.map((sphere, i) => i === index ? sphere.map((v, j) => j === field ? value : v) as Sphere : sphere));
  }, []);
  const requestBody = useMemo(() => ({ method, mode, component, spheres }), [method, mode, component, spheres]);
  const requestKey = useMemo(() => JSON.stringify(requestBody), [requestBody]);
  const visibleData = data && data.method === method && data.mode === mode && data.component === component ? data : null;
  const recordVisualTiming = useCallback((timing: Partial<Timings>) => setTimings(current => ({ ...current, ...timing })), []);
  useEffect(() => {
    if (LOCAL_BUILD) return;
    let active = true;
    const check = async () => {
      try {
        const response = await fetch("/health", { cache:"no-store" });
        const result = await response.json();
        if (active) setServerOnline(response.ok && result.ok === true);
      } catch {
        if (active) setServerOnline(false);
      }
    };
    check();
    const timer = window.setInterval(check, 10000);
    return () => { active = false; clearInterval(timer); };
  }, []);
  useEffect(() => {
    const requestId = ++requestSequence.current;
    const controller = new AbortController();
    setEigenvalue(null);
    setEigenBusy(false);
    setTimings(EMPTY_TIMINGS);
    const timer = window.setTimeout(async () => {
      setBusy(true); setStatus(method === "fd_laplacian" ? "Checking Ritz availability…" : "Evaluating selected mode…");
      try {
        const requestStarted = performance.now();
        const response = await fetch("/infer", { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/octet-stream" }, body: requestKey, signal: controller.signal });
        const binary = await response.arrayBuffer();
        const responseReceived = performance.now();
        if (!response.ok) {
          let message = "Inference failed";
          try { message = JSON.parse(new TextDecoder().decode(binary)).error || message; } catch { /* keep fallback */ }
          throw new Error(message);
        }
        const parseStarted = performance.now();
        const result = parseInferenceBinary(binary, requestId, requestKey);
        const parseMs = performance.now() - parseStarted;
        if (requestId !== requestSequence.current) return;
        setData(result);
        setTimings({ inference: result.elapsed_ms, network: Math.max(0, responseReceived - requestStarted - result.elapsed_ms), parse: parseMs, isosurface: null, render: null });
        setStatus("Ready");
      } catch (error) {
        if ((error as Error).name !== "AbortError" && requestId === requestSequence.current) { setStatus((error as Error).message); if (method === "fd_laplacian") setData(null); }
      } finally { if (requestId === requestSequence.current) setBusy(false); }
    }, 500);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [requestBody, requestKey, method]);

  useEffect(() => {
    if (!data || data.request_key !== requestKey || data.method !== method || data.mode !== mode) return;
    const controller = new AbortController();
    let secondFrame = 0;
    const firstFrame = requestAnimationFrame(() => {
      secondFrame = requestAnimationFrame(async () => {
        setEigenBusy(true);
        try {
          const response = await fetch("/eigenvalue", { method:"POST", headers:{ "Content-Type":"application/json" }, body:JSON.stringify(requestBody), signal:controller.signal });
          const result = await response.json();
          if (!response.ok) throw new Error(result.error || "Eigenvalue estimation failed");
          if (!controller.signal.aborted) setEigenvalue(result.eigenvalue);
        } catch (error) {
          if ((error as Error).name !== "AbortError") setEigenvalue(null);
        } finally {
          if (!controller.signal.aborted) setEigenBusy(false);
        }
      });
    });
    return () => { cancelAnimationFrame(firstFrame); cancelAnimationFrame(secondFrame); controller.abort(); };
  }, [data, method, mode, requestBody, requestKey]);

  return <main>
    <header className="masthead">
      <div className="brand"><span className="mark">E</span><div><strong>EIGENFLUID / METABALL</strong><small>K = 2,048 transfer basis observatory</small></div>{!LOCAL_BUILD && <div className={`server-health ${serverOnline === null ? "checking" : serverOnline ? "online" : "offline"}`}><i />{serverOnline === null ? "Checking inference server" : serverOnline ? "Inference server available" : "Inference server NOT available"}</div>}</div>
      <nav><a href="#explorer">Explorer</a><a href="#archive">Archive</a><a href="/paper/isovorticity_3d_30frames.mp4">Paper video ↗</a>{!LOCAL_BUILD && <a href="https://github.com/baiming-zhang/eigenfluid-metaball-observatory/releases/download/local-v1/eigenfluid-local-inference-windows.zip">Local inference ↓</a>}</nav>
    </header>

    <section className="hero">
      <h1>Shape the domain.<br/><i>Watch the basis respond.</i></h1>
    </section>

    <section id="explorer" className="workspace">
      <div className="explorer-toolbar">
        <div className="toolbar-methods">
          <div className="section-label">01 / METHOD</div>
          <div className="method-list">{METHODS.map(item => <button key={item.id} className={method === item.id ? "selected" : ""} onClick={() => setMethod(item.id)}><span>{item.short}</span><div><b>{item.name}</b><small>{item.meta}</small></div><em>{method === item.id ? liveEigenvalue : "λ live"}</em></button>)}</div>
        </div>
        <div className="toolbar-views">
          <div className="section-label">VIEW</div>
          <div className="display-options">
            <label><input type="checkbox" checked={solidObstacle} onChange={event => setSolidObstacle(event.target.checked)} /><span><b>Opaque metaball obstacle</b><small>smooth implicit envelope</small></span></label>
            <label><input type="checkbox" checked={surfaceModes} onChange={event => setSurfaceModes(event.target.checked)} /><span><b>Vortex mode surface</b><small>checkpoint-derived isosurface</small></span></label>
            <button className="boundary-cycle" onClick={() => setBoundaryMode(current => current === "mask" ? "wire" : current === "wire" ? "hidden" : "mask")}><span className="boundary-state">{boundaryMode === "mask" ? "M" : boundaryMode === "wire" ? "W" : "Ø"}</span><span><b>Outer boundary</b><small>{boundaryMode === "mask" ? "six-face mask" : boundaryMode === "wire" ? "rounded gray wire" : "boundary hidden"}</small></span></button>
          </div>
        </div>
      </div>
      <aside className="controls">
        <div className="section-label mode-head"><span>02 / MODE</span><output>{String(mode + 1).padStart(4, "0")}</output></div>
        <input className="mode-slider" type="range" min="0" max="2047" value={mode} onChange={e => setMode(Number(e.target.value))} />
        <div className="ticks"><span>1</span><span>512</span><span>1024</span><span>1536</span><span>2048</span></div>
        <label className="select-label">Displayed field<select value={component} onChange={e => setComponent(e.target.value)}><option value="magnitude">Velocity magnitude</option><option value="x">Velocity x</option><option value="y">Velocity y</option><option value="z">Velocity z</option></select></label>

        <div className="section-label geometry-head"><span>03 / GEOMETRY</span><button onClick={() => setSpheres(INITIAL)}>reset</button></div>
        {spheres.map((sphere, index) => <div className="sphere-control" key={index}><h3><span>0{index + 1}</span> Metaball {index + 1}</h3><div className="range-grid"><Range label="x" value={sphere[0]} min={0.25} max={0.75} step={0.01} onChange={v => updateSphere(index, 0, v)} /><Range label="y" value={sphere[1]} min={0.25} max={0.75} step={0.01} onChange={v => updateSphere(index, 1, v)} /><Range label="z" value={sphere[2]} min={0.25} max={0.75} step={0.01} onChange={v => updateSphere(index, 2, v)} /><Range label="r" value={sphere[3]} min={0.10} max={0.30} step={0.01} onChange={v => updateSphere(index, 3, v)} /></div></div>)}
      </aside>

      <div className="visuals">
          <div className="visual-head"><div><span className="method-symbol">{active.short}</span><h2>{active.name} <b>mode {mode + 1}</b></h2></div><div className={busy ? "status busy" : "status"}><i />{timings.inference === null ? <span>{status}</span> : <><span className="inference-time">Inference Time · {timings.inference.toFixed(0)} ms</span>{!LOCAL_BUILD && <span className="pipeline-times">Network {timings.network?.toFixed(0) ?? "…"} ms · Parse {timings.parse?.toFixed(1) ?? "…"} ms · Isosurface {timings.isosurface?.toFixed(0) ?? "…"} ms · Render {timings.render?.toFixed(0) ?? "…"} ms</span>}</>}</div></div>
          <FieldScene data={visibleData} spheres={spheres} solidObstacle={solidObstacle} surfaceModes={surfaceModes} boundaryMode={boundaryMode} onTiming={recordVisualTiming} />
        <div className="legend vector-legend"><span className="legend-line mode-line"/>mode / vector field<span className="legend-line obstacle-line"/>obstacle<span className="legend-line boundary-line"/>outer boundary<small>SVG vector slices</small></div>
          <div className="slice-row"><VectorSlice axis="x" vectors={visibleData?.vector_slices.x} spheres={spheres}/><VectorSlice axis="y" vectors={visibleData?.vector_slices.y} spheres={spheres}/><VectorSlice axis="z" vectors={visibleData?.vector_slices.z} spheres={spheres}/></div>
      </div>
    </section>

    <section id="archive" className="archive">
      <div><div className="section-label">04 / REPRODUCIBLE ARCHIVE</div><div className="archive-kicker">GEOMETRY-CONDITIONED VECTOR MODES · STRICT FP32</div><h2>Paper-ready evidence,<br/>without terabytes of cache.</h2><p className="archive-intro">Interact with three overlapping metaballs and inspect any of 2,048 divergence-free modes. The live views evaluate the original epoch-2000 checkpoints on the requested geometry.</p></div>
      <div className="archive-grid"><article><b>Neural transfer</b><p>Three epoch-2000 checkpoints, full training protocols, histories, completion audits and the exact selected-mode inference implementation.</p><a href="/MANIFEST.json">Open manifest ↗</a></article><article><b>Strict FD–Laplacian</b><p>2,048 scalar shapes × 3 polarizations, one 6,144-dimensional global Ritz solve retaining 2,048 complete vector modes.</p><span>Geometry-specific offline solve</span></article><article><b>Temporal comparison</b><p>Thirty-frame GT + four-method videos, three orthogonal slices, framewise errors, PNG and vector PDF endpoints.</p><a href="/paper/isovorticity_3d_30frames.mp4">Play comparison ↗</a></article></div>
    </section>
  </main>;
}
