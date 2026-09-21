# Sine-Gordon MPSKit

This project computes sine-Gordon vacuum states, soliton and breather excitations, soliton propagation, scattering, and continuum scans using the MPSKit library. The Julia code is located in `src/` with dedicated `scripts/` to generate the desired data. The Python scripts in `analysis/` generate figures from the separately distributed TSV data used in [Arxiv](https://arxiv.org/abs/2609.21846) and provided separately via [Zenodo](https://doi.org/10.5281/zenodo.22812679).

## Julia Setup and Calculations

Use Julia 1.12 from this directory and install the locked dependencies.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`Project.toml` lists direct dependencies, while `Manifest.toml` records the tested versions. Run an entry point with `julia --project=. scripts/<name>.jl`. Available scripts are `continuum_scan.jl`, `soliton_free.jl`, and `soliton_scattering.jl`. The scan scripts accept a one-based task index to run sub-tasks exclusively, for example `julia --project=. scripts/continuum_scan.jl 1`. Run options and model parameters are defined in each script or read from its environment variables. New calculation output is stored under `data/`, and logs are written to `logs/`.

`src/sine_gordon.jl` loads the source files in the required order. For interactive use, start Julia with `--project=.` and run `include("src/sine_gordon.jl")`.

## Analysis Figures

Use Python and install the plotting dependencies from this directory.

```bash
python -m pip install -r analysis/requirements.txt
```

Extract the data archive into this project root. The scripts expect the folder structure `data/continuum_scan/`, `data/soliton_free/`, and `data/soliton_scattering/`. If the files are elsewhere, pass `--data-root PATH` to each script.

```bash
python analysis/scattering_plots.py
python analysis/scattering_analysis.py
python analysis/continuum_scan.py
```

`scattering_plots.py` uses scattering run `20260807` and its parameter-matched free run. Select another run with `--run RUN_DIRECTORY`. `scattering_analysis.py` compares four scattering runs with their free references. `continuum_scan.py` selects the latest dated continuum run. Pin a run with `--run 20260617` if other continuum runs are present.

Figures are saved to `analysis/figures/<script>/`. Use `--figure-dir PATH` to change the destination. Pass `--no-usetex` to any script when LaTeX is unavailable.
