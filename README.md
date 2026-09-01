# 3D Metaball K=2048 Results + Basis Observatory

This directory is a compact, paper-oriented package of the latest K=2048
Potential, Velocity, Vorticity and strict global-Ritz FD-Laplacian results.
It includes the final checkpoints, training protocols and histories, exact
selected-mode inference code, paper videos/figures, and the minimum FD-Lap
global transform needed to audit the retained 2,048-dimensional subspace.

## Run the interactive HTML

Double-click `start_observatory.ps1`, or run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_observatory.ps1
```

Then open <http://127.0.0.1:8780>. Geometry sliders use the original trained
xyzr ranges. Potential, Velocity and Vorticity evaluate the actual epoch-2000
checkpoint. The compact preview grid defaults to 64³ and can be changed before
launch with `$env:METABALL_PREVIEW_GRID=48`.

Strict FD-Laplacian is not a neural transfer model: an exact basis on a new
geometry requires rebuilding the discrete operators and running the global
6,144→2,048 Ritz solve. The dashboard therefore preserves its exact archived
result and labels it offline rather than substituting a surrogate.

## Contents

- `models/`: three final checkpoints plus completion/protocol/history records.
- `results_archive/fd_laplacian/`: global Ritz transform, retained Gram and audits.
- `backend/inference_server.py`: exact checkpoint reader and selected-mode server.
- `reproducibility/`: native CUDA trainer/reconstructor, configs, and Ritz builder.
- `public/paper/`: directly viewable MP4 and PNG results.
- `paper_assets/`: vector PDFs and compact temporal metrics.
- `public/MANIFEST.json`: file roles, contracts and SHA-256 checksums.

Full 128³ basis chunks, 6144-candidate spatial caches, Adam moments and duplicate
checkpoints are intentionally excluded; they are large intermediate artifacts,
not required for the packaged paper figures or neural selected-mode inference.

## Public deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for Docker and native Windows deployment.
The public server includes per-IP request limits, bounded inference concurrency,
request-size limits, a health endpoint, and delayed cached Rayleigh estimates.
