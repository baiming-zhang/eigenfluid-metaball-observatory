"""Exact selected-mode inference for the K=2048 geometry-conditioned SIREN models.

The server reads the original TVFP32V2 checkpoint, evaluates one complete
antisymmetric potential A=(A12,A13,A23), applies the trained double-zero
boundary envelope, and forms the divergence-free velocity u=div(A) with
centered finite differences on a compact preview grid.
"""

from __future__ import annotations

import json
import math
import mimetypes
import mmap
import os
import struct
import time
from collections import OrderedDict, defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "potential": ROOT / "models" / "potential" / "epoch_2000.bin",
    "velocity": ROOT / "models" / "velocity" / "epoch_2000.bin",
    "vorticity": ROOT / "models" / "vorticity" / "epoch_2000.bin",
}
LAYERS = ((15, 32, 0, 480), (32, 32, 512, 1536), (32, 32, 1568, 2592),
          (32, 32, 2624, 3648), (32, 3, 3680, 3776))
PARAMETERS_PER_MODE = 3779
HEADER = struct.Struct("<8s8I")
GRID_N = int(os.environ.get("METABALL_PREVIEW_GRID", "64"))
CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
FIELD_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
MAX_BODY_BYTES = int(os.environ.get("METABALL_MAX_BODY_BYTES", "65536"))
MAX_CONCURRENT = int(os.environ.get("METABALL_MAX_CONCURRENT", "2"))
RATE_LIMIT = int(os.environ.get("METABALL_RATE_LIMIT", "30"))
RATE_WINDOW = float(os.environ.get("METABALL_RATE_WINDOW", "60"))
ALLOWED_ORIGINS = {item.strip() for item in os.environ.get("METABALL_ALLOWED_ORIGINS", "*").split(",") if item.strip()}
INFERENCE_SLOTS = BoundedSemaphore(max(1, MAX_CONCURRENT))
RATE_BUCKETS: defaultdict[str, deque[float]] = defaultdict(deque)
RATE_LOCK = Lock()


def checkpoint_header(path: Path) -> dict[str, int | str]:
    with path.open("rb") as stream:
        raw = stream.read(HEADER.size)
    magic, version, epoch, modes, ppm, raw_inputs, components, jets, scalar_bytes = HEADER.unpack(raw)
    if magic.rstrip(b"\0") != b"TVFP32V2" or ppm != PARAMETERS_PER_MODE:
        raise ValueError(f"unsupported checkpoint contract: {path}")
    return {
        "magic": magic.rstrip(b"\0").decode(), "version": version, "epoch": epoch,
        "modes": modes, "parameters_per_mode": ppm, "raw_inputs": raw_inputs,
        "components": components, "jet_components": jets, "scalar_bytes": scalar_bytes,
    }


HEADERS = {name: checkpoint_header(path) for name, path in MODELS.items() if path.exists()}


def load_mode(path: Path, mode: int) -> np.ndarray:
    header = checkpoint_header(path)
    if mode < 0 or mode >= int(header["modes"]):
        raise ValueError("mode must be in [0, 2047]")
    offset = HEADER.size + mode * PARAMETERS_PER_MODE * 4
    with path.open("rb") as stream:
        mm = mmap.mmap(stream.fileno(), length=offset + PARAMETERS_PER_MODE * 4, access=mmap.ACCESS_READ)
        result = np.frombuffer(mm, dtype="<f4", count=PARAMETERS_PER_MODE, offset=offset).copy()
        mm.close()
    return result


def validate_geometry(spheres: np.ndarray) -> None:
    if spheres.shape != (3, 4) or not np.isfinite(spheres).all():
        raise ValueError("geometry must contain three finite [x,y,z,r] spheres")
    if np.any(spheres[:, :3] < 0.25) or np.any(spheres[:, :3] > 0.75):
        raise ValueError("sphere centers must remain in the trained interval [0.25, 0.75]")
    if np.any(spheres[:, 3] < 0.10) or np.any(spheres[:, 3] > 0.30):
        raise ValueError("sphere radii must remain in the trained interval [0.10, 0.30]")
    links = np.zeros((3, 3), dtype=bool)
    for a in range(3):
        for b in range(a + 1, 3):
            overlap = spheres[a, 3] + spheres[b, 3] - np.linalg.norm(spheres[a, :3] - spheres[b, :3])
            links[a, b] = links[b, a] = overlap >= 0.025
    seen = {0}
    for _ in range(3):
        seen |= {b for a in tuple(seen) for b in range(3) if links[a, b]}
    if len(seen) != 3:
        raise ValueError("the three metaballs must form a connected geometry (minimum overlap 0.025)")


def obstacle_level(points: np.ndarray, spheres: np.ndarray) -> np.ndarray:
    distances = np.stack([
        np.linalg.norm(points - sphere[:3], axis=1) - sphere[3] for sphere in spheres
    ], axis=1)
    minimum = distances.min(axis=1)
    shifted = np.exp(-(distances - minimum[:, None]) / 0.02)
    return minimum - 0.02 * np.log(shifted.sum(axis=1))


def envelope(points: np.ndarray, spheres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    outer = 1.0 - np.sum(((points - 0.5) / 0.49) ** 8, axis=1)
    obstacle = obstacle_level(points, spheres)
    fluid = (outer > 0.0) & (obstacle > 0.0)
    value = np.zeros(points.shape[0], dtype=np.float32)
    safe_outer = outer[fluid] / (0.18 + outer[fluid])
    safe_obstacle = obstacle[fluid] / (0.10 + obstacle[fluid])
    value[fluid] = (safe_outer * safe_obstacle) ** 2
    return value, fluid


def forward_raw(points: np.ndarray, spheres: np.ndarray, parameter: np.ndarray) -> np.ndarray:
    features = np.empty((points.shape[0], 15), dtype=np.float32)
    features[:, :3] = 2.0 * points - 1.0
    cursor = 3
    for sphere in spheres:
        features[:, cursor:cursor + 3] = 2.0 * ((sphere[:3] - 0.25) / 0.50) - 1.0
        features[:, cursor + 3] = 2.0 * ((sphere[3] - 0.10) / 0.20) - 1.0
        cursor += 4
    state = features
    for layer_index, (inputs, outputs, weight_offset, bias_offset) in enumerate(LAYERS):
        weight = parameter[weight_offset:weight_offset + inputs * outputs].reshape((outputs, inputs), order="F")
        bias = parameter[bias_offset:bias_offset + outputs]
        state = state @ weight.T + bias
        if layer_index < 4:
            state = np.sin(state * (6.0 if layer_index == 0 else 36.0))
    return state.astype(np.float32, copy=False)


def infer(payload: dict[str, Any]) -> dict[str, Any]:
    method = str(payload.get("method", "potential"))
    if method == "fd_laplacian":
        raise RuntimeError("FD-Laplacian is a geometry-specific global Ritz solve, not a learned geometry map; use reproduce_fd_laplacian.ps1 for an exact new-geometry basis.")
    if method not in MODELS or not MODELS[method].exists():
        raise ValueError(f"unknown or unpacked method: {method}")
    mode = int(payload.get("mode", 0))
    component = str(payload.get("component", "magnitude"))
    if component not in {"magnitude", "x", "y", "z"}:
        raise ValueError("component must be magnitude, x, y, or z")
    spheres = np.asarray(payload.get("spheres"), dtype=np.float32)
    validate_geometry(spheres)
    cache_key = json.dumps([method, mode, component, np.round(spheres, 4).tolist()], separators=(",", ":"))
    if cache_key in CACHE:
        CACHE.move_to_end(cache_key)
        if cache_key in FIELD_CACHE:
            FIELD_CACHE.move_to_end(cache_key)
        return CACHE[cache_key]

    axis = np.linspace(0.0, 1.0, GRID_N, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)
    env, fluid = envelope(points, spheres)
    parameter = load_mode(MODELS[method], mode)
    inference_started = time.perf_counter()
    raw = forward_raw(points, spheres, parameter)
    elapsed_ms = (time.perf_counter() - inference_started) * 1000.0
    potential = (raw * env[:, None]).reshape((GRID_N, GRID_N, GRID_N, 3))
    h = float(axis[1] - axis[0])
    a12, a13, a23 = potential[..., 0], potential[..., 1], potential[..., 2]
    velocity = np.empty_like(potential)
    velocity[..., 0] = np.gradient(a12, h, axis=1) + np.gradient(a13, h, axis=2)
    velocity[..., 1] = -np.gradient(a12, h, axis=0) + np.gradient(a23, h, axis=2)
    velocity[..., 2] = -np.gradient(a13, h, axis=0) - np.gradient(a23, h, axis=1)
    fluid3 = fluid.reshape((GRID_N, GRID_N, GRID_N))
    velocity[~fluid3] = 0.0
    if component == "magnitude":
        scalar = np.linalg.norm(velocity, axis=-1)
    else:
        scalar = velocity[..., {"x": 0, "y": 1, "z": 2}[component]]
    scale = float(np.percentile(np.abs(scalar[fluid3]), 99.0)) if np.any(fluid3) else 1.0
    scale = max(scale, 1e-8)
    normalized = np.clip(scalar / scale, -1.0, 1.0)
    center = GRID_N // 2
    slices = {
        "x": normalized[center, :, :].astype(np.float16).tolist(),
        "y": normalized[:, center, :].astype(np.float16).tolist(),
        "z": normalized[:, :, center].astype(np.float16).tolist(),
    }
    vector_scale = float(np.percentile(np.linalg.norm(velocity[fluid3], axis=-1), 99.0)) if np.any(fluid3) else 1.0
    vector_scale = max(vector_scale, 1e-8)
    normalized_velocity = np.clip(velocity / vector_scale, -1.0, 1.0)
    vector_slices = {
        "x": normalized_velocity[center, :, :, :].astype(np.float16).tolist(),
        "y": normalized_velocity[:, center, :, :].astype(np.float16).tolist(),
        "z": normalized_velocity[:, :, center, :].astype(np.float16).tolist(),
    }
    flat_score = np.abs(normalized).ravel()
    valid_indices = np.flatnonzero(fluid)
    count = min(1800, valid_indices.size)
    selected = valid_indices[np.argpartition(flat_score[valid_indices], -count)[-count:]] if count else np.array([], dtype=int)
    cloud = np.column_stack((points[selected], normalized.ravel()[selected])).astype(np.float32)
    response = {
        "method": method, "mode": mode, "component": component, "grid": GRID_N,
        "scale": scale, "elapsed_ms": round(elapsed_ms, 1), "fluid_fraction": round(float(fluid.mean()), 4),
        "cloud": cloud, "slices": slices, "vector_slices": vector_slices,
        "volume": normalized.astype(np.float32).ravel(),
        "contract": "TVFP32V2 · exact checkpoint forward · finite-difference preview of u=div(A)",
    }
    FIELD_CACHE[cache_key] = {
        "method": method, "potential": potential, "velocity": velocity,
        "fluid": fluid3, "spacing": h,
    }
    CACHE[cache_key] = response
    while len(CACHE) > 8:
        expired_key, _ = CACHE.popitem(last=False)
        FIELD_CACHE.pop(expired_key, None)
    return response


def encode_inference_binary(response: dict[str, Any]) -> bytes:
    """Pack metadata plus aligned, little-endian FP32 field arrays."""
    metadata_keys = (
        "method", "mode", "component", "grid", "scale", "elapsed_ms",
        "fluid_fraction", "contract",
    )
    header = {key: response[key] for key in metadata_keys}
    descriptors: dict[str, dict[str, Any]] = {}
    chunks: list[bytes] = []
    offset = 0

    def append_array(name: str, value: Any) -> None:
        nonlocal offset
        array = np.ascontiguousarray(value, dtype="<f4")
        payload = array.tobytes(order="C")
        descriptors[name] = {"offset": offset, "length": int(array.size), "shape": list(array.shape)}
        chunks.append(payload)
        offset += len(payload)

    append_array("cloud", response["cloud"])
    append_array("volume", response["volume"])
    for axis in ("x", "y", "z"):
        append_array(f"slices.{axis}", response["slices"][axis])
    for axis in ("x", "y", "z"):
        append_array(f"vector_slices.{axis}", response["vector_slices"][axis])

    header["format"] = "eigenfluid-fp32-v1"
    header["arrays"] = descriptors
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    prefix = struct.pack("<I", len(header_bytes)) + header_bytes
    padding = b"\0" * ((-len(prefix)) % 4)
    return prefix + padding + b"".join(chunks)


def estimate_eigenvalue(payload: dict[str, Any]) -> dict[str, Any]:
    method = str(payload.get("method", "potential"))
    mode = int(payload.get("mode", 0))
    component = str(payload.get("component", "magnitude"))
    spheres = np.asarray(payload.get("spheres"), dtype=np.float32)
    validate_geometry(spheres)
    cache_key = json.dumps([method, mode, component, np.round(spheres, 4).tolist()], separators=(",", ":"))
    if cache_key not in FIELD_CACHE:
        CACHE.pop(cache_key, None)
        infer(payload)
    record = FIELD_CACHE[cache_key]
    FIELD_CACHE.move_to_end(cache_key)
    started = time.perf_counter()
    if method == "potential":
        field = record["potential"]
    elif method == "velocity":
        field = record["velocity"]
    elif method == "vorticity":
        velocity = record["velocity"]
        h = record["spacing"]
        field = np.empty_like(velocity)
        field[..., 0] = np.gradient(velocity[..., 2], h, axis=1) - np.gradient(velocity[..., 1], h, axis=2)
        field[..., 1] = np.gradient(velocity[..., 0], h, axis=2) - np.gradient(velocity[..., 2], h, axis=0)
        field[..., 2] = np.gradient(velocity[..., 1], h, axis=0) - np.gradient(velocity[..., 0], h, axis=1)
    else:
        raise ValueError("eigenvalue estimate is available only for neural methods")
    fluid = record["fluid"]
    h = record["spacing"]
    denominator = float(np.sum(field[fluid] ** 2, dtype=np.float64))
    numerator = 0.0
    for component_index in range(3):
        for axis_index in range(3):
            derivative = np.gradient(field[..., component_index], h, axis=axis_index)
            numerator += float(np.sum(derivative[fluid] ** 2, dtype=np.float64))
    value = numerator / max(denominator, 1e-20)
    return {
        "eigenvalue": value,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
        "contract": "post-render discrete Rayleigh quotient on the cached 64^3 field",
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "MetaballBasis/1.0"

    def _headers(self, status: int = 200, content_type: str = "application/json; charset=utf-8", content_length: int | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        origin = self.headers.get("Origin")
        if "*" in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", "*")
        elif origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("X-Content-Type-Options", "nosniff")
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))
        self.end_headers()

    def _client_ip(self) -> str:
        return self.headers.get("CF-Connecting-IP") or self.headers.get("X-Forwarded-For", "").split(",")[0].strip() or self.client_address[0]

    def _within_rate_limit(self) -> bool:
        now = time.monotonic()
        with RATE_LOCK:
            bucket = RATE_BUCKETS[self._client_ip()]
            while bucket and now - bucket[0] >= RATE_WINDOW:
                bucket.popleft()
            if len(bucket) >= RATE_LIMIT:
                return False
            bucket.append(now)
            return True

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._headers(204)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._headers()
            body = {"ok": len(HEADERS) == 3, "models": HEADERS, "grid": GRID_N}
            self.wfile.write(json.dumps(body).encode()); return
        relative = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        host = self.headers.get("Host", "").lower()
        local_host = host.startswith("127.0.0.1") or host.startswith("localhost") or host.startswith("[::1]")
        local_root = ROOT / "out-local"
        static_root = (local_root if local_host and local_root.exists() else ROOT / "out").resolve()
        candidate = (static_root / relative).resolve()
        if static_root not in candidate.parents and candidate != static_root:
            self.send_error(403); return
        if candidate.is_dir():
            candidate = candidate / "index.html"
        if not candidate.exists():
            self.send_error(404); return
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(200); self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(candidate.stat().st_size)); self.end_headers()
        with candidate.open("rb") as stream:
            self.wfile.write(stream.read())

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/infer", "/eigenvalue"}:
            self._headers(404); self.wfile.write(b'{"error":"not found"}'); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                self._headers(413); self.wfile.write(b'{"error":"request body too large or empty"}'); return
            if not self._within_rate_limit():
                self._headers(429); self.wfile.write(b'{"error":"rate limit exceeded"}'); return
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not INFERENCE_SLOTS.acquire(blocking=False):
                self._headers(503); self.wfile.write(b'{"error":"server busy; retry shortly"}'); return
            try:
                body = infer(payload) if self.path == "/infer" else estimate_eigenvalue(payload)
            finally:
                INFERENCE_SLOTS.release()
            if self.path == "/infer":
                encoded = encode_inference_binary(body)
                self._headers(content_type="application/octet-stream", content_length=len(encoded))
                self.wfile.write(encoded)
                return
            self._headers(); self.wfile.write(json.dumps(body, separators=(",", ":")).encode())
        except RuntimeError as exc:
            self._headers(409); self.wfile.write(json.dumps({"error": str(exc), "offline": True}).encode())
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._headers(422); self.wfile.write(json.dumps({"error": str(exc)}).encode())
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            self._headers(500); self.wfile.write(json.dumps({"error": f"inference failed: {exc}"}).encode())

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)


if __name__ == "__main__":
    missing = [str(path) for path in MODELS.values() if not path.exists()]
    if missing:
        raise SystemExit("missing packaged checkpoints: " + ", ".join(missing))
    address = (os.environ.get("METABALL_BIND_HOST", "127.0.0.1"), int(os.environ.get("METABALL_API_PORT", "8780")))
    print(f"Metaball basis observatory at http://{address[0]}:{address[1]}", flush=True)
    ThreadingHTTPServer(address, Handler).serve_forever()
