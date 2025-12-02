# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains interactive research demonstrations built with [marimo](https://marimo.io), a reactive Python notebook framework. Demos are deployed to GitHub Pages at https://kermodegroup.github.io/demos

## Repository Structure

- `apps/` - Marimo applications (exported in "run" mode, code hidden)
- `apps/lib/` - Refactored library modules (for local development and testing)
- `notebooks/` - Marimo notebooks (exported in "edit" mode, code visible) - currently empty
- `scripts/build.py` - Build script that exports notebooks to HTML-WASM

## Commands

### Run a demo locally
```bash
marimo run apps/regression-demo.py
```

### Edit a demo locally (opens in browser with live reload)
```bash
marimo edit apps/regression-demo.py
```

### Build all demos for deployment
```bash
python scripts/build.py
```
Output goes to `_site/` directory.

### Build with lib/ sync check
```bash
python scripts/build.py --sync-lib
```
Bundles lib/ modules with marimo files and checks sync status before export.

### Check lib/ sync status only (no build)
```bash
python scripts/build.py --check-sync
```

## Architecture

- **Marimo apps** use inline script dependencies (PEP 723 format) at the top of each `.py` file - no separate requirements.txt
- **Build process** exports each `.py` file to standalone HTML-WASM:
  - Files in `apps/` are exported with `--mode run --no-show-code` (app mode)
  - Files in `notebooks/` are exported with `--mode edit` (notebook mode)
- **Deployment** is automatic via GitHub Actions on push to main

## Refactored Library Modules (`apps/lib/`)

The `apps/lib/` directory contains refactored, modular versions of the regression demo code. These modules are the canonical source for the ML/scientific logic and can be used for:
- Unit testing
- Reuse in other demos
- Cleaner development workflow

### Module Overview

| Module | Description |
|--------|-------------|
| `models.py` | `MyBayesianRidge`, `ConformalPrediction`, `NeuralNetworkRegression`, `QuantileRegressionUQ` |
| `kernels.py` | GP kernels (RBF, Matérn, bump, polynomial) and prediction functions |
| `basis.py` | Basis feature functions (RBF, Fourier, LJ, custom) |
| `metrics.py` | Probabilistic metrics (log likelihood, CRPS) |
| `data.py` | Ground truth functions and data generation |
| `optimization.py` | GP hyperparameter optimization |

### WASM Export and Sync Workflow

The marimo WASM export bundles only a single `.py` file. The `regression-demo.py` contains inline copies of the lib/ code for WASM compatibility.

**When modifying core logic:**
1. Update the lib/ module first (source of truth)
2. Run `python scripts/build.py --check-sync` to verify sync status
3. Manually sync changes to inline definitions in `regression-demo.py`
4. The CI build uses `--sync-lib` to validate sync before export

**Build script options:**
- `--sync-lib`: Bundles lib/ with marimo files, warns if out of sync
- `--check-sync`: Quick sync status check without building

### Example: Using lib/ locally

```python
from lib import MyBayesianRidge, fit_gp_numpy, crps_gaussian
from lib import ground_truth, make_rbf_features
```
