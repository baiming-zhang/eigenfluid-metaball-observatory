"""Reconstruct three DNS frames with 1024/2048 distinct scalar Laplacian modes.

Each scalar eigenfunction supplies exactly one antisymmetric tensor-potential
mode.  Tensor components cycle A12, A13, A23, but no scalar spatial shape is
reused.  Thus the K=1024 basis is the strict prefix of the K=2048 basis.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
from scipy.ndimage import map_coordinates
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_SCRIPT = PROJECT_ROOT / "scripts" / "reconstruct_independent_dns_gt_full128.py"
RENDER_SCRIPT = PROJECT_ROOT / "scripts" / "render_three_dns_reconstruction.py"
DEFAULT_SCALAR_ROOT = PROJECT_ROOT / "results" / "laplacian_128_scalar_k2048"
DEFAULT_OLD_CELL_CACHE = (
    PROJECT_ROOT
    / "results"
    / "independent_dns_gt_128_reconstruction"
    / "laplacian_scalar_cell_center_128.npy"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "reconstruction_three_dns_laplacian_scalar_k1024_k2048_full128"
)

K_SMALL = 1024
K_LARGE = 2048
BASE_SCALAR_MODES = 342
CASES = (
    ("case_01_independent_dns", "Independent DNS", "independent_dns_gt_128"),
    ("case_02_large_eddy", "Large-eddy TG/ABC DNS", "independent_dns_large_eddy_gt_128"),
    (
        "case_03_natural_large_eddy",
        "Natural low-wave DNS",
        "independent_dns_natural_large_eddy_gt_128",
    ),
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


full = load_module("lap_scalar_2048_full", FULL_SCRIPT)
render = load_module("lap_scalar_2048_render", RENDER_SCRIPT)
helper = full.helper
base = full.base
training = full.training


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def block_ranges(count: int, blocks: int = 2) -> list[tuple[int, int]]:
    edges = np.linspace(0, count, blocks + 1, dtype=np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(blocks)]


def relative_l2(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    a = np.asarray(reference[mask], dtype=np.float64)
    b = np.asarray(estimate[mask], dtype=np.float64)
    return float(np.linalg.norm(b - a) / np.linalg.norm(a))


def build_cell_scalar_cache(
    cfg: Any,
    geometry: np.ndarray,
    endpoint_path: Path,
    old_cache_path: Path,
    output_path: Path,
    progress_path: Path,
) -> np.memmap:
    endpoint_grid = base.build_grid(cfg, 128, geometry)
    endpoint = np.load(endpoint_path, mmap_mode="r")
    if endpoint.shape != (len(endpoint_grid["points"]), K_LARGE):
        raise ValueError(f"unexpected endpoint scalar cache {endpoint.shape}")
    old_cache = np.load(old_cache_path, mmap_mode="r")
    if old_cache.shape != (128**3, BASE_SCALAR_MODES):
        raise ValueError(f"unexpected old cell scalar cache {old_cache.shape}")
    if output_path.exists():
        output = np.load(output_path, mmap_mode="r+")
        if output.shape != (128**3, K_LARGE):
            raise ValueError(f"unexpected cell scalar output {output.shape}")
    else:
        output = open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(128**3, K_LARGE),
        )
    next_mode = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            log("reuse complete 2048-scalar cell-center cache")
            return np.load(output_path, mmap_mode="r")
        next_mode = int(progress.get("next_scalar_mode", 0))
    if next_mode < BASE_SCALAR_MODES:
        log("copy exact cell-centered scalar modes 1-342")
        for start in range(0, 128**3, 131_072):
            stop = min(start + 131_072, 128**3)
            output[start:stop, :BASE_SCALAR_MODES] = old_cache[start:stop]
        output.flush()
        next_mode = BASE_SCALAR_MODES
        atomic_json(
            progress_path,
            {
                "complete": False,
                "next_scalar_mode": next_mode,
                "scalar_modes": K_LARGE,
                "grid": [128, 128, 128],
            },
        )

    coordinates = (np.arange(128, dtype=np.float64) + 0.5) / 128
    query_1d = coordinates * 127
    query_z, query_y, query_x = np.meshgrid(
        query_1d, query_1d, query_1d, indexing="ij"
    )
    query = np.stack((query_z.ravel(), query_y.ravel(), query_x.ravel()))
    for mode in range(next_mode, K_LARGE):
        endpoint_volume = np.zeros(endpoint_grid["shape"], dtype=np.float32)
        endpoint_volume[endpoint_grid["mask"]] = np.asarray(
            endpoint[:, mode], dtype=np.float32
        )
        output[:, mode] = map_coordinates(
            endpoint_volume,
            query,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float32)
        if (mode + 1) % 10 == 0 or mode + 1 == K_LARGE:
            output.flush()
            atomic_json(
                progress_path,
                {
                    "complete": mode + 1 == K_LARGE,
                    "next_scalar_mode": mode + 1,
                    "scalar_modes": K_LARGE,
                    "grid": [128, 128, 128],
                    "endpoint_interpolation": "trilinear endpoint-to-cell-center",
                },
            )
            log(f"cell-centered scalar modes {mode + 1}/{K_LARGE}")
    return np.load(output_path, mmap_mode="r")


def build_one_to_one_velocity_basis(
    scalar_path: Path,
    mask: np.ndarray,
    flat_indices: np.ndarray,
    output_path: Path,
    progress_path: Path,
    scalar_chunk: int,
    spatial_blocks: int = 2,
) -> np.memmap:
    expected = (len(flat_indices), 3, K_LARGE)
    if output_path.exists():
        output = np.load(output_path, mmap_mode="r+")
        if output.shape != expected or output.dtype != np.float32:
            raise ValueError(f"unexpected velocity basis {output.shape} {output.dtype}")
    else:
        output = open_memmap(output_path, mode="w+", dtype=np.float32, shape=expected)
    next_block = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            log("reuse complete one-to-one scalar K=2048 velocity basis")
            return np.load(output_path, mmap_mode="r")
        next_block = int(progress.get("next_block", 0))
    scalar = np.load(scalar_path, mmap_mode="r")
    if scalar.shape != (mask.size, K_LARGE):
        raise ValueError(f"unexpected scalar cache {scalar.shape}")
    ranges = block_ranges(len(flat_indices), spatial_blocks)
    spacing = 1.0 / 128
    for block_index, (start, stop) in enumerate(ranges):
        if block_index < next_block:
            continue
        indices = flat_indices[start:stop]
        block = np.zeros((stop - start, 3, K_LARGE), dtype=np.float32)
        log(
            f"build unique-scalar vector block {block_index + 1}/{spatial_blocks}: "
            f"{start}:{stop}, {block.nbytes / 1024**3:.2f} GiB"
        )
        for scalar_start in range(0, K_LARGE, scalar_chunk):
            scalar_stop = min(scalar_start + scalar_chunk, K_LARGE)
            width = scalar_stop - scalar_start
            volume = np.ascontiguousarray(
                scalar[:, scalar_start:scalar_stop], dtype=np.float32
            ).reshape(mask.shape + (width,))
            derivative_z, derivative_y, derivative_x = np.gradient(
                volume, spacing, axis=(0, 1, 2), edge_order=2
            )
            dz = derivative_z.reshape(mask.size, width)[indices]
            dy = derivative_y.reshape(mask.size, width)[indices]
            dx = derivative_x.reshape(mask.size, width)[indices]
            u = np.zeros((stop - start, width), dtype=np.float32)
            v = np.zeros_like(u)
            w = np.zeros_like(u)
            modes = np.arange(scalar_start, scalar_stop)
            a12 = np.flatnonzero(modes % 3 == 0)
            a13 = np.flatnonzero(modes % 3 == 1)
            a23 = np.flatnonzero(modes % 3 == 2)
            u[:, a12] = dy[:, a12]
            v[:, a12] = -dx[:, a12]
            u[:, a13] = dz[:, a13]
            w[:, a13] = -dx[:, a13]
            v[:, a23] = dz[:, a23]
            w[:, a23] = -dy[:, a23]
            block[:, 0, scalar_start:scalar_stop] = u
            block[:, 1, scalar_start:scalar_stop] = v
            block[:, 2, scalar_start:scalar_stop] = w
            del volume, derivative_z, derivative_y, derivative_x, dz, dy, dx, u, v, w
            if scalar_stop % 256 == 0 or scalar_stop == K_LARGE:
                log(
                    f"vector block {block_index + 1}/{spatial_blocks} differentiated "
                    f"{scalar_stop}/{K_LARGE} unique scalar shapes"
                )
        output[start:stop] = block
        output.flush()
        del block
        atomic_json(
            progress_path,
            {
                "complete": block_index + 1 == spatial_blocks,
                "next_block": block_index + 1,
                "spatial_blocks": spatial_blocks,
                "grid": [128, 128, 128],
                "scalar_modes": K_LARGE,
                "vector_modes": K_LARGE,
                "one_to_one": True,
                "orientation": "mode mod 3 cycles A12, A13, A23",
                "spatial_shape_reuse": False,
            },
        )
        log(
            f"unique-scalar velocity basis block "
            f"{block_index + 1}/{spatial_blocks} complete"
        )
    return np.load(output_path, mmap_mode="r")


def mass_and_correlations(
    basis_path: Path,
    targets: np.ndarray,
    output_path: Path,
    progress_path: Path,
    spatial_blocks: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as cached:
            mass = np.asarray(cached["mass"], dtype=np.float64)
            correlations = np.asarray(cached["correlations"], dtype=np.float64)
        if mass.shape == (K_LARGE, K_LARGE) and correlations.shape == (
            len(CASES),
            K_LARGE,
        ):
            log("reuse K=2048 mass matrix and correlations")
            return mass, correlations
    basis = np.load(basis_path, mmap_mode="r")
    mass = np.zeros((K_LARGE, K_LARGE), dtype=np.float64)
    correlations = np.zeros((len(CASES), K_LARGE), dtype=np.float64)
    next_block = 0
    partial_path = output_path.with_name(output_path.stem + "_partial.npz")
    if progress_path.exists() and partial_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        next_block = int(progress.get("next_block", 0))
        with np.load(partial_path, allow_pickle=False) as cached:
            mass = np.asarray(cached["mass"], dtype=np.float64)
            correlations = np.asarray(cached["correlations"], dtype=np.float64)
    if training.DEVICE.type != "cuda":
        raise RuntimeError("the two-block full-128 Gram accumulation requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    weight = (1.0 / 128) ** 3
    for block_index, (start, stop) in enumerate(
        block_ranges(len(targets[0]), spatial_blocks)
    ):
        if block_index < next_block:
            continue
        gib = (stop - start) * 3 * K_LARGE * 4 / 1024**3
        log(
            f"GPU Gram block {block_index + 1}/{spatial_blocks}: "
            f"{gib:.2f} GiB basis"
        )
        values = torch.as_tensor(
            np.asarray(basis[start:stop]), device=training.DEVICE, dtype=torch.float32
        )
        flat = values.reshape(-1, K_LARGE)
        mass += (weight * (flat.T @ flat)).double().cpu().numpy()
        target_tensor = torch.as_tensor(
            np.asarray(targets[:, start:stop]),
            device=training.DEVICE,
            dtype=torch.float32,
        )
        correlations += (
            weight * torch.einsum("nck,tnc->tk", values, target_tensor)
        ).double().cpu().numpy()
        del target_tensor, flat, values
        torch.cuda.empty_cache()
        atomic_npz(partial_path, mass=mass, correlations=correlations)
        atomic_json(
            progress_path,
            {
                "complete": block_index + 1 == spatial_blocks,
                "next_block": block_index + 1,
                "spatial_blocks": spatial_blocks,
                "tf32": False,
                "grid": [128, 128, 128],
            },
        )
    mass = 0.5 * (mass + mass.T)
    atomic_npz(output_path, mass=mass, correlations=correlations)
    return mass, correlations


def reconstruct_pair(
    basis_path: Path,
    coefficients_1024: np.ndarray,
    coefficients_2048: np.ndarray,
    mask: np.ndarray,
    flat_indices: np.ndarray,
    output_1024: Path,
    output_2048: Path,
    progress_path: Path,
) -> tuple[np.memmap, np.memmap]:
    shape = (len(CASES), 128, 128, 128, 3)
    if output_1024.exists() and output_2048.exists():
        small = np.load(output_1024, mmap_mode="r+")
        large = np.load(output_2048, mmap_mode="r+")
    else:
        small = open_memmap(output_1024, mode="w+", dtype=np.float32, shape=shape)
        large = open_memmap(output_2048, mode="w+", dtype=np.float32, shape=shape)
        small[:] = 0
        large[:] = 0
        small.flush()
        large.flush()
    next_block = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            return np.load(output_1024, mmap_mode="r"), np.load(output_2048, mmap_mode="r")
        next_block = int(progress.get("next_block", 0))
    basis = np.load(basis_path, mmap_mode="r")
    coeff_small = torch.as_tensor(
        coefficients_1024, device=training.DEVICE, dtype=torch.float32
    )
    coeff_large = torch.as_tensor(
        coefficients_2048, device=training.DEVICE, dtype=torch.float32
    )
    for block_index, (start, stop) in enumerate(block_ranges(len(flat_indices), 2)):
        if block_index < next_block:
            continue
        values = torch.as_tensor(
            np.asarray(basis[start:stop]), device=training.DEVICE, dtype=torch.float32
        )
        recon_large = torch.einsum("nck,tk->tnc", values, coeff_large).cpu().numpy()
        recon_small = torch.einsum(
            "nck,tk->tnc", values[:, :, :K_SMALL], coeff_small
        ).cpu().numpy()
        indices = flat_indices[start:stop]
        for case in range(len(CASES)):
            small[case].reshape(mask.size, 3)[indices] = recon_small[case]
            large[case].reshape(mask.size, 3)[indices] = recon_large[case]
        small.flush()
        large.flush()
        del values, recon_small, recon_large
        torch.cuda.empty_cache()
        atomic_json(
            progress_path,
            {
                "complete": block_index + 1 == 2,
                "next_block": block_index + 1,
                "spatial_blocks": 2,
                "grid": [128, 128, 128],
                "modes": [K_SMALL, K_LARGE],
            },
        )
        log(f"reconstructed K=1024 and K=2048 block {block_index + 1}/2")
    return np.load(output_1024, mmap_mode="r"), np.load(output_2048, mmap_mode="r")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--scalar-root", type=Path, default=DEFAULT_SCALAR_ROOT)
    parser.add_argument("--old-cell-cache", type=Path, default=DEFAULT_OLD_CELL_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scalar-chunk", type=int, default=64)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    scalar_root = args.scalar_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    basis_dir = scalar_root / "basis_full128_one_to_one"
    basis_dir.mkdir(parents=True, exist_ok=True)

    scalar_metadata = json.loads(
        (scalar_root / "laplacian_scalar_metadata_128.json").read_text(encoding="utf-8")
    )
    if not scalar_metadata.get("complete") or scalar_metadata.get("scalar_modes") != K_LARGE:
        raise RuntimeError("the 2048-distinct-scalar spectrum is not complete")
    config = json.loads((workspace / "configs" / "winner_k64_e2000.json").read_text(encoding="utf-8"))
    cfg = base.training.Config(**config)
    base.training.validate_contract(cfg)
    geometry = base.training.geometry_from_token(cfg, 12_000_001)

    reference_mask: np.ndarray | None = None
    targets: list[np.ndarray] = []
    gt_vorticity: list[np.ndarray] = []
    for _, _, source_name in CASES:
        source = workspace / "results" / source_name
        mask = np.load(source / "fluid_mask.npy").astype(bool, copy=False)
        if reference_mask is None:
            reference_mask = mask
        elif not np.array_equal(reference_mask, mask):
            raise ValueError("the three DNS masks differ")
        case_geometry = np.load(source / "geometry_xyzr.npy")
        if not np.allclose(case_geometry, geometry, rtol=0.0, atol=1.0e-7):
            raise ValueError(f"{source_name} geometry differs")
        velocity = np.load(source / "velocity" / "frame_000.npy", mmap_mode="r")
        omega = np.load(source / "vorticity" / "frame_000.npy", mmap_mode="r")
        targets.append(np.asarray(velocity[mask], dtype=np.float32))
        gt_vorticity.append(omega)
    assert reference_mask is not None
    mask = reference_mask
    _, flat_indices = helper.cell_center_points(mask)
    target_stack = np.stack(targets)

    cell_scalar_path = basis_dir / "laplacian_scalar_cell_center_128.npy"
    build_cell_scalar_cache(
        cfg,
        geometry,
        scalar_root / "laplacian_scalar_eigenvectors_128.npy",
        args.old_cell_cache.resolve(),
        cell_scalar_path,
        basis_dir / "laplacian_scalar_cell_center_progress.json",
    )
    basis_path = basis_dir / "raw_velocity_128.npy"
    build_one_to_one_velocity_basis(
        cell_scalar_path,
        mask,
        flat_indices,
        basis_path,
        basis_dir / "raw_velocity_progress.json",
        args.scalar_chunk,
    )
    mass, correlations = mass_and_correlations(
        basis_path,
        target_stack,
        basis_dir / "mass_and_correlations.npz",
        basis_dir / "mass_and_correlations.progress.json",
    )
    mass_1024 = mass[:K_SMALL, :K_SMALL]
    correlations_1024 = correlations[:, :K_SMALL]
    eig_2048 = np.linalg.eigvalsh(mass)
    eig_1024 = np.linalg.eigvalsh(mass_1024)
    for label, values in (("K=1024", eig_1024), ("K=2048", eig_2048)):
        if values[0] <= values[-1] * 1.0e-10:
            raise FloatingPointError(
                f"{label} mass matrix rank failure: {values[0]:.3e}/{values[-1]:.3e}"
            )
    coefficients_1024 = np.linalg.solve(mass_1024, correlations_1024.T).T
    coefficients_2048 = np.linalg.solve(mass, correlations.T).T
    normal_1024 = float(
        np.linalg.norm(coefficients_1024 @ mass_1024 - correlations_1024)
        / np.linalg.norm(correlations_1024)
    )
    normal_2048 = float(
        np.linalg.norm(coefficients_2048 @ mass - correlations)
        / np.linalg.norm(correlations)
    )
    atomic_npz(
        output / "coefficients_scalar_k1024_k2048.npz",
        coefficients_k1024=coefficients_1024.astype(np.float32),
        coefficients_k2048=coefficients_2048.astype(np.float32),
    )
    velocity_1024, velocity_2048 = reconstruct_pair(
        basis_path,
        coefficients_1024.astype(np.float32),
        coefficients_2048.astype(np.float32),
        mask,
        flat_indices,
        output / "laplacian_scalar_k1024_velocity_three_cases.npy",
        output / "laplacian_scalar_k2048_velocity_three_cases.npy",
        output / "reconstruction.progress.json",
    )
    omega_1024 = helper.curl_trajectory(
        velocity_1024,
        mask,
        output / "laplacian_scalar_k1024_vorticity_three_cases.npy",
    )
    omega_2048 = helper.curl_trajectory(
        velocity_2048,
        mask,
        output / "laplacian_scalar_k2048_vorticity_three_cases.npy",
    )

    artifacts: list[Path] = [
        output / "coefficients_scalar_k1024_k2048.npz",
        output / "laplacian_scalar_k1024_velocity_three_cases.npy",
        output / "laplacian_scalar_k2048_velocity_three_cases.npy",
        output / "laplacian_scalar_k1024_vorticity_three_cases.npy",
        output / "laplacian_scalar_k2048_vorticity_three_cases.npy",
    ]
    metrics: list[dict[str, Any]] = []
    surfaces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, float]] = {}
    for index, (case_id, label, source_name) in enumerate(CASES):
        source = workspace / "results" / source_name
        gt_velocity = np.load(source / "velocity" / "frame_000.npy", mmap_mode="r")
        gt_omega = gt_vorticity[index]
        velocity_error_1024 = relative_l2(gt_velocity, velocity_1024[index], mask)
        velocity_error_2048 = relative_l2(gt_velocity, velocity_2048[index], mask)
        omega_error_1024 = relative_l2(gt_omega, omega_1024[index], mask)
        omega_error_2048 = relative_l2(gt_omega, omega_2048[index], mask)
        case_dir = output / case_id
        case_dir.mkdir(exist_ok=True)
        paths = (
            case_dir / "velocity_k1024_128.npy",
            case_dir / "velocity_k2048_128.npy",
            case_dir / "vorticity_k1024_128.npy",
            case_dir / "vorticity_k2048_128.npy",
            case_dir / "coefficients_k1024.npy",
            case_dir / "coefficients_k2048.npy",
        )
        arrays = (
            velocity_1024[index],
            velocity_2048[index],
            omega_1024[index],
            omega_2048[index],
            coefficients_1024[index].astype(np.float32),
            coefficients_2048[index].astype(np.float32),
        )
        for path, array in zip(paths, arrays):
            np.save(path, array)
        artifacts.extend(paths)

        fields = {
            "gt": np.linalg.norm(np.asarray(gt_omega), axis=-1).astype(np.float32),
            "k1024": np.linalg.norm(np.asarray(omega_1024[index]), axis=-1).astype(np.float32),
            "k2048": np.linalg.norm(np.asarray(omega_2048[index]), axis=-1).astype(np.float32),
        }
        for key, values in fields.items():
            values[~mask] = 0
            surfaces[(case_id, key)] = render.surface(values)
        figure = plt.figure(figsize=(12.0, 4.2), facecolor="white")
        for column, (key, title, color) in enumerate(
            (
                ("gt", "DNS ground truth", render.DARK),
                ("k1024", "1024 distinct scalar modes", render.BLUE),
                ("k2048", "2048 distinct scalar modes", render.RED),
            )
        ):
            axis = figure.add_subplot(1, 3, column + 1, projection="3d")
            mesh_data = surfaces[(case_id, key)]
            render.draw(axis, mesh_data, color)
            axis.set_title(f"{title}\n|vorticity| iso {mesh_data[2]:.3g}", fontsize=10, color=render.DARK)
        figure.suptitle(
            f"{label} - one scalar shape per mode - full 128^3 projection\n"
            f"velocity rel. L2: K1024 {velocity_error_1024:.2%}, K2048 {velocity_error_2048:.2%}",
            fontsize=12,
            color=render.DARK,
            y=0.99,
        )
        figure.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.84, wspace=0.005)
        artifacts.extend(render.save_figure(figure, case_dir / "gt_k1024_k2048_vorticity_isosurfaces"))
        metrics.append(
            {
                "id": case_id,
                "label": label,
                "frame": 0,
                "k1024_scalar_modes": K_SMALL,
                "k2048_scalar_modes": K_LARGE,
                "k1024_velocity_relative_l2": velocity_error_1024,
                "k2048_velocity_relative_l2": velocity_error_2048,
                "k1024_vorticity_relative_l2": omega_error_1024,
                "k2048_vorticity_relative_l2": omega_error_2048,
                "velocity_error_reduction_fraction": 1.0 - velocity_error_2048 / velocity_error_1024,
            }
        )

    figure = plt.figure(figsize=(12.0, 11.2), facecolor="white")
    for row, (case_id, label, _) in enumerate(CASES):
        for column, (key, title, color) in enumerate(
            (
                ("gt", "DNS ground truth", render.DARK),
                ("k1024", "1024 scalar modes", render.BLUE),
                ("k2048", "2048 scalar modes", render.RED),
            )
        ):
            axis = figure.add_subplot(3, 3, row * 3 + column + 1, projection="3d")
            mesh_data = surfaces[(case_id, key)]
            render.draw(axis, mesh_data, color)
            axis.set_title(f"{label}\n{title} - iso {mesh_data[2]:.3g}", fontsize=9, color=render.DARK, pad=0)
    figure.suptitle(
        "Laplacian reconstruction - 1024 vs 2048 distinct scalar modes - full 128^3",
        fontsize=13,
        color=render.DARK,
        y=0.995,
    )
    figure.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.94, wspace=0.002, hspace=0.08)
    artifacts.extend(render.save_figure(figure, output / "three_dns_scalar_k1024_k2048_full128"))

    csv_path = output / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    artifacts.append(csv_path)
    completion = {
        "complete": all(path.is_file() and path.stat().st_size > 0 for path in artifacts),
        "method": "one distinct scalar Dirichlet-Laplacian eigenfunction per tensor-potential mode",
        "scalar_modes": [K_SMALL, K_LARGE],
        "vector_modes": [K_SMALL, K_LARGE],
        "k1024_is_prefix_of_k2048": True,
        "tensor_orientation": "mode mod 3 cycles A12, A13, A23",
        "spatial_shape_reuse": False,
        "grid": [128, 128, 128],
        "fluid_nodes": int(len(flat_indices)),
        "projection": "all fluid nodes",
        "subsampling": False,
        "spatial_blocks": 2,
        "tf32": False,
        "mass_condition_number_k1024": float(eig_1024[-1] / eig_1024[0]),
        "mass_condition_number_k2048": float(eig_2048[-1] / eig_2048[0]),
        "normal_equation_relative_residual_k1024": normal_1024,
        "normal_equation_relative_residual_k2048": normal_2048,
        "cases": metrics,
        "basis": str(basis_path),
        "artifacts": [str(path) for path in artifacts],
    }
    atomic_json(output / "completion.json", completion)
    rows = [
        "# Laplacian scalar-mode reconstruction: K=1024 versus K=2048",
        "",
        "- One distinct scalar eigenfunction per mode; no spatial shape is reused.",
        "- K=1024 is the strict prefix of K=2048.",
        "- Full 128 x 128 x 128 projection on all fluid nodes; no subsampling.",
        "",
        "| Case | K1024 velocity rel. L2 | K2048 velocity rel. L2 | Reduction | K1024 vorticity rel. L2 | K2048 vorticity rel. L2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        rows.append(
            f"| {item['label']} | {item['k1024_velocity_relative_l2']:.4%} | "
            f"{item['k2048_velocity_relative_l2']:.4%} | "
            f"{item['velocity_error_reduction_fraction']:.2%} | "
            f"{item['k1024_vorticity_relative_l2']:.4%} | "
            f"{item['k2048_vorticity_relative_l2']:.4%} |"
        )
    (output / "REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    if not completion["complete"]:
        raise RuntimeError("scalar K1024/K2048 artifact validation failed")
    log(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
