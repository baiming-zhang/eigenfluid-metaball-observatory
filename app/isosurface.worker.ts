/// <reference lib="webworker" />

import * as THREE from "three";
import { MarchingCubes } from "three/examples/jsm/objects/MarchingCubes.js";

type Sphere = [number, number, number, number];
type IsoGeometry = { positions: Float32Array; normals: Float32Array };
type WorkerRequest = { mode: { values: ArrayBuffer; resolution: number } | null; obstacle: { spheres: Sphere[]; resolution: number } | null };

function extractSurface(values: Float32Array, resolution: number, isolation: number, absolute: boolean): IsoGeometry {
  const surface = new MarchingCubes(resolution, new THREE.MeshBasicMaterial(), false, false, 240000);
  surface.isolation = isolation;
  for (let x = 0; x < resolution; x++) for (let y = 0; y < resolution; y++) for (let z = 0; z < resolution; z++) {
    const index = (x * resolution + y) * resolution + z;
    const value = values[index] ?? 0;
    surface.setCell(x, y, z, absolute ? Math.abs(value) : value);
  }
  surface.update();
  const count = surface.geometry.drawRange.count;
  const position = surface.geometry.getAttribute("position");
  const normal = surface.geometry.getAttribute("normal");
  const positions = new Float32Array(count * 3);
  const normals = new Float32Array(count * 3);
  positions.set((position.array as Float32Array).subarray(0, count * 3));
  normals.set((normal.array as Float32Array).subarray(0, count * 3));
  surface.geometry.dispose();
  (surface.material as THREE.Material).dispose();
  return { positions, normals };
}

function obstacleField(spheres: Sphere[], resolution: number) {
  const values = new Float32Array(resolution ** 3);
  for (let x = 0; x < resolution; x++) for (let y = 0; y < resolution; y++) for (let z = 0; z < resolution; z++) {
    const px = x / (resolution - 1), py = y / (resolution - 1), pz = z / (resolution - 1);
    let field = 0;
    for (const [sx, sy, sz, radius] of spheres) field += Math.exp((radius - Math.hypot(px - sx, py - sy, pz - sz)) / 0.032);
    values[(x * resolution + y) * resolution + z] = field;
  }
  return values;
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const started = performance.now();
  const mode = event.data.mode ? extractSurface(new Float32Array(event.data.mode.values), event.data.mode.resolution, 0.42, true) : null;
  const obstacle = event.data.obstacle ? extractSurface(obstacleField(event.data.obstacle.spheres, event.data.obstacle.resolution), event.data.obstacle.resolution, 1, false) : null;
  const transfer: Transferable[] = [];
  if (mode) transfer.push(mode.positions.buffer, mode.normals.buffer);
  if (obstacle) transfer.push(obstacle.positions.buffer, obstacle.normals.buffer);
  self.postMessage({ mode, obstacle, elapsedMs: performance.now() - started }, transfer);
};

export {};
