"""Generate and gate an independent envelope-matched full-128^3 GT frame."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import fft
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_SCRIPT = PROJECT_ROOT / "scripts" / "reconstruct_independent_dns_gt_full128.py"
FIRST_SCRIPT = PROJECT_ROOT / "scripts" / "reconstruct_first_frame_full128.py"
SCAN_SCRIPT = PROJECT_ROOT / "scripts" / "reconstruct_large_eddy_first_frame_full128.py"
DEFAULT_BASIS = PROJECT_ROOT / "results" / "independent_dns_gt_128_reconstruction_full128"
DEFAULT_GEOMETRY = PROJECT_ROOT / "results" / "independent_dns_gt_128"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "envelope_matched_natural_gt_first_frame_128"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "results" / "k1024_e2000" / "run_001" / "checkpoints" / "final.pt"
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


full = load_module("envelope_gt_full128_core", FULL_SCRIPT)
first = load_module("envelope_gt_first_frame_core", FIRST_SCRIPT)
scan = load_module("envelope_gt_projection_core", SCAN_SCRIPT)
training = full.training
helper = full.helper


def natural_low_wave_potential(
    resolution: int,
    seed: int,
    minimum_wave_number: float,
    maximum_wave_number: float,
    peak_wave_number: float,
    spectral_width: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((resolution, resolution, resolution, 3), dtype=np.float32)
    spectrum = fft.fftn(noise, axes=(0, 1, 2), workers=-1)
    frequency = np.fft.fftfreq(resolution, d=1.0 / resolution)
    kz, ky, kx = np.meshgrid(frequency, frequency, frequency, indexing="ij")
    radius = np.sqrt(kx * kx + ky * ky + kz * kz)
    support = (radius >= minimum_wave_number) & (radius <= maximum_wave_number)
    amplitude = np.zeros_like(radius, dtype=np.float32)
    amplitude[support] = (
        np.exp(
            -0.5 * ((radius[support] - peak_wave_number) / spectral_width) ** 2
        )
        / np.maximum(radius[support], 1.0) ** 1.5
    )
    potential = fft.ifftn(
        spectrum * amplitude[..., None], axes=(0, 1, 2), workers=-1
    ).real.astype(np.float32)
    return potential


def exact_training_envelope(
    cfg: Any,
    points: np.ndarray,
    geometry: np.ndarray,
    flat_indices: np.ndarray,
    shape: tuple[int, int, int],
    batch_size: int = 65536,
) -> np.ndarray:
    result = np.zeros(int(np.prod(shape)), dtype=np.float32)
    geometry_tensor = torch.as_tensor(
        geometry, device=training.DEVICE, dtype=training.DTYPE
    )
    with torch.inference_mode():
        for start in range(0, len(points), batch_size):
            stop = min(start + batch_size, len(points))
            point_tensor = torch.as_tensor(
                points[start:stop], device=training.DEVICE, dtype=training.DTYPE
            )
            envelope, _ = training.envelope_and_gradient(
                cfg, point_tensor, geometry_tensor
            )
            result[flat_indices[start:stop]] = envelope.cpu().numpy()
    return result.reshape(shape)


def velocity_from_potential(potential: np.ndarray, spacing: float) -> np.ndarray:
    a12 = potential[..., 0]
    a13 = potential[..., 1]
    a23 = potential[..., 2]
    _, derivative_y_12, derivative_x_12 = np.gradient(
        a12, spacing, edge_order=2
    )
    derivative_z_13, _, derivative_x_13 = np.gradient(
        a13, spacing, edge_order=2
    )
    derivative_z_23, derivative_y_23, _ = np.gradient(
        a23, spacing, edge_order=2
    )
    return np.stack(
        (
            derivative_y_12 + derivative_z_13,
            -derivative_x_12 + derivative_z_23,
            -derivative_x_13 - derivative_y_23,
        ),
        axis=3,
    ).astype(np.float32)


def curl(field: np.ndarray, spacing: float) -> np.ndarray:
    derivative_z, derivative_y, derivative_x = np.gradient(
        field, spacing, axis=(0, 1, 2), edge_order=2
    )
    return np.stack(
        (
            derivative_y[..., 2] - derivative_z[..., 1],
            derivative_z[..., 0] - derivative_x[..., 2],
            derivative_x[..., 1] - derivative_y[..., 0],
        ),
        axis=3,
    ).astype(np.float32)


def divergence_relative_l2(field: np.ndarray, spacing: float) -> float:
    derivative_x = np.gradient(field[..., 0], spacing, axis=2, edge_order=2)
    derivative_y = np.gradient(field[..., 1], spacing, axis=1, edge_order=2)
    derivative_z = np.gradient(field[..., 2], spacing, axis=0, edge_order=2)
    divergence = derivative_x + derivative_y + derivative_z
    gradient_scale = np.sqrt(
        sum(
            np.sum(np.gradient(field[..., component], spacing, axis=axis) ** 2)
            for component in range(3)
            for axis in range(3)
        )
    )
    return float(np.linalg.norm(divergence) / max(float(gradient_scale), 1.0e-30))


def run(
    basis_dir: Path,
    geometry_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    protocol = full.Full128Config()
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = np.load(geometry_dir / "fluid_mask.npy")
    obstacle_mask = np.load(geometry_dir / "obstacle_mask.npy")
    points, flat_indices = helper.cell_center_points(mask)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = training.Config(**checkpoint["config"])
    geometry = training.geometry_from_token(cfg, protocol.geometry_token)
    potential_raw = natural_low_wave_potential(
        protocol.resolution,
        seed,
        minimum_wave_number=1.0,
        maximum_wave_number=3.0,
        peak_wave_number=1.5,
        spectral_width=0.8,
    )
    envelope = exact_training_envelope(
        cfg, points, geometry, flat_indices, mask.shape
    )
    potential = potential_raw * envelope[..., None]
    velocity = velocity_from_potential(potential, 1.0 / protocol.resolution)
    unmasked_divergence = divergence_relative_l2(
        velocity, 1.0 / protocol.resolution
    )
    velocity[~mask] = 0.0
    rms = float(np.sqrt(np.mean(np.sum(velocity[mask].astype(np.float64) ** 2, axis=1))))
    scale = 0.32 / rms
    potential *= scale
    velocity *= scale
    vorticity = curl(velocity, 1.0 / protocol.resolution)
    vorticity[~mask] = 0.0
    full.atomic_npy(output_dir / "gt_raw_potential.npy", potential_raw)
    full.atomic_npy(output_dir / "training_envelope.npy", envelope)
    full.atomic_npy(output_dir / "gt_potential.npy", potential)
    full.atomic_npy(output_dir / "gt_velocity.npy", velocity)
    full.atomic_npy(output_dir / "gt_vorticity.npy", vorticity)

    target = np.asarray(velocity[mask], dtype=np.float32)[None]
    eigen_dir = basis_dir / "basis" / "eigenfluid"
    lap_dir = basis_dir / "basis" / "laplacian"
    eigen_correlations = scan.correlations(
        eigen_dir / "raw_velocity_128.npy", target, protocol, "EigenFluid envelope GT"
    )[0]
    lap_correlations = scan.correlations(
        lap_dir / "raw_velocity_128.npy", target, protocol, "Laplacian envelope GT"
    )[0]
    with np.load(eigen_dir / "first_frame_mass_projection.npz", allow_pickle=False) as cached:
        eigen_mass = cached["mass"]
    with np.load(lap_dir / "first_frame_mass_projection.npz", allow_pickle=False) as cached:
        lap_mass = cached["mass"]
    eigen_coefficients = np.linalg.solve(eigen_mass, eigen_correlations)
    lap_coefficients = np.linalg.solve(lap_mass, lap_correlations)
    eigen_velocity = first.reconstruct(
        eigen_dir / "raw_velocity_128.npy",
        eigen_coefficients,
        mask,
        flat_indices,
        output_dir / "eigenfluid_velocity.npy",
        protocol,
        "EigenFluid envelope GT",
    )
    lap_velocity = first.reconstruct(
        lap_dir / "raw_velocity_128.npy",
        lap_coefficients,
        mask,
        flat_indices,
        output_dir / "laplacian_velocity.npy",
        protocol,
        "Laplacian envelope GT",
    )
    eigen_vorticity = curl(np.asarray(eigen_velocity), 1.0 / protocol.resolution)
    eigen_vorticity[~mask] = 0.0
    lap_vorticity = curl(np.asarray(lap_velocity), 1.0 / protocol.resolution)
    lap_vorticity[~mask] = 0.0
    full.atomic_npy(output_dir / "eigenfluid_vorticity.npy", eigen_vorticity)
    full.atomic_npy(output_dir / "laplacian_vorticity.npy", lap_vorticity)
    errors = {
        "eigenfluid_velocity_relative_l2": first.relative_l2(
            velocity[mask], eigen_velocity[mask]
        ),
        "laplacian_velocity_relative_l2": first.relative_l2(
            velocity[mask], lap_velocity[mask]
        ),
        "eigenfluid_vorticity_relative_l2": first.relative_l2(
            vorticity[mask], eigen_vorticity[mask]
        ),
        "laplacian_vorticity_relative_l2": first.relative_l2(
            vorticity[mask], lap_vorticity[mask]
        ),
    }
    png, pdf, levels = first.render(
        vorticity,
        eigen_vorticity,
        lap_vorticity,
        obstacle_mask,
        mask,
        {
            "eigenfluid": errors["eigenfluid_velocity_relative_l2"],
            "laplacian": errors["laplacian_velocity_relative_l2"],
        },
        output_dir,
        protocol,
        gt_label="Independent envelope-matched GT",
        figure_title="Full-128³ first-frame reconstruction — shared GT isovorticity levels",
    )
    gate_passed = bool(
        errors["eigenfluid_velocity_relative_l2"] < 0.10
        and errors["laplacian_velocity_relative_l2"] < 0.10
    )
    result = {
        "complete": True,
        "gate_passed": gate_passed,
        "continue_30_frames": gate_passed,
        "grid": [128, 128, 128],
        "fluid_nodes": int(np.count_nonzero(mask)),
        "modes": {"eigenfluid": 1024, "laplacian": 1024},
        "gt_independent_of_modal_bases": True,
        "raw_potential_spectrum": {
            "minimum_wave_number": 1.0,
            "maximum_wave_number": 3.0,
            "peak_wave_number": 1.5,
            "seed": seed,
        },
        "boundary": {
            "kind": "exact EigenFluid training antisymmetric-potential envelope",
            "outer_transition": float(cfg.outer_envelope_transition),
            "obstacle_transition": float(cfg.obstacle_envelope_transition),
        },
        "velocity_operator": "cell-centered 128^3 numpy.gradient, second order",
        "velocity_rms": float(np.sqrt(np.mean(np.sum(velocity[mask].astype(np.float64) ** 2, axis=1)))),
        "discrete_divergence_relative_l2_before_solid_fill": unmasked_divergence,
        "fluid_mask_interface_divergence_relative_l2_after_solid_fill": divergence_relative_l2(
            velocity, 1.0 / protocol.resolution
        ),
        "errors": errors,
        "shared_gt_isovorticity_levels": list(levels),
        "png": str(png.resolve()),
        "pdf": str(pdf.resolve()),
    }
    full.atomic_json(output_dir / "completion.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis-dir", type=Path, default=DEFAULT_BASIS)
    parser.add_argument("--geometry-dir", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20_260_828)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.basis_dir.resolve(),
        args.geometry_dir.resolve(),
        args.checkpoint.resolve(),
        args.output_dir.resolve(),
        args.seed,
    )


if __name__ == "__main__":
    main()
