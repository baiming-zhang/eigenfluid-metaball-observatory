"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { MarchingCubes } from "three/examples/jsm/objects/MarchingCubes.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";

type Method = "potential" | "velocity" | "vorticity" | "fd_laplacian";
type Sphere = [number, number, number, number];
type BoundaryMode = "mask" | "wire" | "hidden";
type Inference = {
  method: Method;
  mode: number;
  component: string;
  grid: number;
  scale: number;
  elapsed_ms: number;
  fluid_fraction: number;
  cloud: number[][];
  volume: number[];
  slices: Record<"x" | "y" | "z", number[][]>;
  vector_slices: Record<"x" | "y" | "z", number[][][]>;
  contract: string;
};

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

function Slice({ axis, values, spheres }: { axis: "x" | "y" | "z"; values?: number[][]; spheres: Sphere[] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !values?.length) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const n = values.length;
    canvas.width = n;
    canvas.height = n;
    const image = ctx.createImageData(n, n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const v = Math.max(-1, Math.min(1, values[n - 1 - y][x]));
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

function VectorSlice({ axis, vectors, spheres }: { axis: "x" | "y" | "z"; vectors?: number[][][]; spheres: Sphere[] }) {
  const boundary = useMemo(() => obstacleContourPath(axis, spheres), [axis, spheres]);
  const arrows = useMemo(() => {
    if (!vectors?.length) return [];
    const n = vectors.length, step = Math.max(1, Math.floor(n / 13));
    const result: { x1: number; y1: number; x2: number; y2: number; opacity: number }[] = [];
    for (let row = Math.floor(step / 2); row < n; row += step) for (let column = Math.floor(step / 2); column < n; column += step) {
      const value = vectors[row]?.[column];
      if (!value) continue;
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

function buildOuterBoundary(mode: Exclude<BoundaryMode, "hidden">) {
  const geometry = new RoundedBoxGeometry(1, 1, 1, 10, 0.16);
  if (mode === "wire") return new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry),
    new THREE.LineBasicMaterial({ color:0x8b9291, transparent:true, opacity:0.48 }),
  );
  return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color:0xd7dad8, transparent:true, opacity:0.18, side:THREE.DoubleSide, depthWrite:false }));
}

function FieldScene({ data, spheres, solidObstacle, surfaceModes, boundaryMode }: { data: Inference | null; spheres: Sphere[]; solidObstacle: boolean; surfaceModes: boolean; boundaryMode: BoundaryMode }) {
  const mount = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const host = mount.current;
    if (!host) return;
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
    if (!solidObstacle) spheres.forEach((s, i) => {
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(s[3], 28, 18),
        new THREE.MeshBasicMaterial({ color: i === 1 ? 0xb73d49 : 0x2e6fa5, transparent: true, opacity: 0.12, wireframe: true }),
      );
      mesh.position.set(s[0] - 0.5, s[1] - 0.5, s[2] - 0.5);
      group.add(mesh);
    });
    if (solidObstacle) group.add(buildMetaballObstacle(spheres));
    if (surfaceModes && data?.volume?.length) {
      group.add(buildIsoSurface(data.volume, data.grid, 0xbd3b48, 0.42, value => Math.abs(value), 0.78));
    }
    if (!surfaceModes && data?.cloud?.length) {
      const positions = new Float32Array(data.cloud.length * 3);
      const colors = new Float32Array(data.cloud.length * 3);
      data.cloud.forEach((p, i) => {
        positions[3 * i] = p[0] - 0.5; positions[3 * i + 1] = p[1] - 0.5; positions[3 * i + 2] = p[2] - 0.5;
        const color = new THREE.Color(tone(p[3]));
        colors[3 * i] = color.r; colors[3 * i + 1] = color.g; colors[3 * i + 2] = color.b;
      });
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
    let frame = 0;
    const draw = () => { frame = requestAnimationFrame(draw); renderer.render(scene, camera); };
    draw();
    const resize = () => { const w = host.clientWidth, h = host.clientHeight; camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h); };
    const observer = new ResizeObserver(resize); observer.observe(host);
    return () => { cancelAnimationFrame(frame); observer.disconnect(); renderer.dispose(); host.removeChild(renderer.domElement); };
  }, [data, spheres, solidObstacle, surfaceModes, boundaryMode]);
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
  const [busy, setBusy] = useState(false);
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
  useEffect(() => {
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
    const controller = new AbortController();
    setEigenvalue(null);
    setEigenBusy(false);
    const timer = window.setTimeout(async () => {
      setBusy(true); setStatus(method === "fd_laplacian" ? "Checking Ritz availability…" : "Evaluating selected mode…");
      try {
        const response = await fetch("/infer", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(requestBody), signal: controller.signal });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Inference failed");
        setData(result); setStatus(`Inference Time · ${result.elapsed_ms.toFixed(0)} ms`);
      } catch (error) {
        if ((error as Error).name !== "AbortError") { setStatus((error as Error).message); if (method === "fd_laplacian") setData(null); }
      } finally { if (!controller.signal.aborted) setBusy(false); }
    }, 280);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [requestBody, method]);

  useEffect(() => {
    if (!data || data.method !== method || data.mode !== mode) return;
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
  }, [data, method, mode, requestBody]);

  return <main>
    <header className="masthead">
      <div className="brand"><span className="mark">E</span><div><strong>EIGENFLUID / METABALL</strong><small>K = 2,048 transfer basis observatory</small></div></div>
      <nav><a href="#explorer">Explorer</a><a href="#archive">Archive</a><a href="/paper/isovorticity_3d_30frames.mp4">Paper video ↗</a></nav>
    </header>

    <section className="hero">
      <h1>Shape the domain.<br/><i>Watch the basis respond.</i></h1>
      <div className={`server-health ${serverOnline === false ? "offline" : ""}`}><i />{serverOnline === null ? "Checking inference" : serverOnline ? "Inference available" : "Inference unavailable"}</div>
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
        <div className="visual-head"><div><span className="method-symbol">{active.short}</span><h2>{active.name} <b>mode {mode + 1}</b></h2></div><div className={busy ? "status busy" : "status"}><i />{status}</div></div>
        <FieldScene data={data} spheres={spheres} solidObstacle={solidObstacle} surfaceModes={surfaceModes} boundaryMode={boundaryMode} />
        <div className="legend vector-legend"><span className="legend-line mode-line"/>mode / vector field<span className="legend-line obstacle-line"/>obstacle<span className="legend-line boundary-line"/>outer boundary<small>SVG vector slices</small></div>
        <div className="slice-row"><VectorSlice axis="x" vectors={data?.vector_slices.x} spheres={spheres}/><VectorSlice axis="y" vectors={data?.vector_slices.y} spheres={spheres}/><VectorSlice axis="z" vectors={data?.vector_slices.z} spheres={spheres}/></div>
      </div>
    </section>

    <section id="archive" className="archive">
      <div><div className="section-label">04 / REPRODUCIBLE ARCHIVE</div><div className="archive-kicker">GEOMETRY-CONDITIONED VECTOR MODES · STRICT FP32</div><h2>Paper-ready evidence,<br/>without terabytes of cache.</h2><p className="archive-intro">Interact with three overlapping metaballs and inspect any of 2,048 divergence-free modes. The live views evaluate the original epoch-2000 checkpoints on the requested geometry.</p></div>
      <div className="archive-grid"><article><b>Neural transfer</b><p>Three epoch-2000 checkpoints, full training protocols, histories, completion audits and the exact selected-mode inference implementation.</p><a href="/MANIFEST.json">Open manifest ↗</a></article><article><b>Strict FD–Laplacian</b><p>2,048 scalar shapes × 3 polarizations, one 6,144-dimensional global Ritz solve retaining 2,048 complete vector modes.</p><span>Geometry-specific offline solve</span></article><article><b>Temporal comparison</b><p>Thirty-frame GT + four-method videos, three orthogonal slices, framewise errors, PNG and vector PDF endpoints.</p><a href="/paper/isovorticity_3d_30frames.mp4">Play comparison ↗</a></article></div>
    </section>
  </main>;
}
