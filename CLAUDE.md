# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains interactive research demonstrations built with [marimo](https://marimo.io), a reactive Python notebook framework. Demos are deployed to https://sciml.warwick.ac.uk with automatic WASM/live server selection based on dependencies.

## Repository Structure

```
demos/
├── apps/                    # WASM-compatible notebooks (static HTML export)
│   └── lib/                 # Shared library modules
├── notebooks/               # Live server notebooks (require native deps)
├── scripts/
│   ├── build.py             # WASM HTML export script
│   ├── categorize_notebooks.py  # WASM vs live detection
│   ├── deploy.sh            # Server-side deployment (systemd + nginx)
│   └── deploy-warwick.sh    # Local deployment script (requires 2FA)
├── 404.html                 # GitHub Pages redirect
└── index.html               # GitHub Pages redirect
```

## Deployment Architecture

### Dual Deployment Modes

Notebooks are automatically categorized and deployed in one of two modes:

1. **WASM (Static)** - Notebooks in `apps/` with pure Python dependencies
   - Exported to HTML+WASM via `marimo export html-wasm`
   - Served as static files from `/var/www/marimo-wasm/`
   - No authentication required
   - Runs entirely in browser via Pyodide

2. **Live (Server)** - Notebooks in `notebooks/` with native dependencies
   - Each notebook runs as a separate marimo process
   - Managed by systemd services (port 2718+)
   - Proxied through nginx with WebSocket support
   - Protected by basic auth

### WASM Incompatible Packages

The `scripts/categorize_notebooks.py` detects these packages and routes to live:
- ML frameworks: `jax`, `torch`, `tensorflow`
- File system: `watchdog`, `psutil`
- Database drivers: `psycopg2`, `mysqlclient`
- Native extensions: `opencv-python`, `cryptography`, `grpcio`, `pyarrow`, etc.

## Commands

### Local Development

```bash
# Run a notebook locally
marimo run apps/regression-demo.py
marimo edit notebooks/jax-test.py

# Test WASM compatibility detection
python scripts/categorize_notebooks.py --offline

# Build WASM notebooks locally
python scripts/build.py --sync-lib --output-dir _wasm_site
```

### Deployment

Deployment requires 2FA authentication to the university server, so it must be done locally:

```bash
# Deploy from local machine (builds WASM, rsyncs, restarts services)
./scripts/deploy-warwick.sh
```

The deploy script:
1. Categorizes notebooks into WASM vs live
2. Builds WASM notebooks to `_wasm_site/`
3. Rsyncs WASM files to `/var/www/marimo-wasm/` on server
4. Rsyncs live notebooks to `/home/svc_user/marimo-server/notebooks/`
5. Restarts marimo service on server

## Server Infrastructure

**Target server:** sciml.warwick.ac.uk

```
/home/ubuntu/marimo-server/
├── app.py           # FastAPI entry point (alternative to systemd)
├── deploy.sh        # Deployment script (symlink to scripts/deploy.sh)
├── notebooks/       # Live notebook .py files
└── .venv/           # Python environment with marimo + deps

/var/www/marimo-wasm/
└── apps/            # WASM HTML files + assets
```

**Services:**
- nginx: SSL termination, basic auth, reverse proxy
- systemd: `marimo-{notebook-name}.service` per live notebook
- Let's Encrypt: SSL certificates

## Migration Checklist

When migrating to a new server:

1. **Server setup:**
   - Install Python 3.12+, nginx, certbot
   - Create marimo server directory
   - Create Python venv with `uv`: `uv venv && uv pip install marimo jax jaxlib`
   - Create `/var/www/marimo-wasm/` with correct permissions

2. **SSL & Auth:**
   - Run `certbot --nginx -d yourdomain.com`
   - Create `/etc/nginx/.htpasswd` with `htpasswd -c /etc/nginx/.htpasswd username`

3. **Deploy script:**
   - Update `DOMAIN` variable in `scripts/deploy.sh`
   - Update paths in `scripts/deploy-warwick.sh`
   - Symlink or copy deploy.sh to server

4. **DNS:**
   - Point domain to new server IP

5. **GitHub Pages redirects:**
   - Update URLs in `404.html` and `index.html` if domain changes

## Marimo Notebook Conventions

- Include dependencies in PEP 723 script metadata at top of file
- Avoid `watchdog` and other dev-only deps that break WASM
- Use `mo.ui.*` for interactive elements
- Last expression in cell is auto-displayed
- For matplotlib: use `plt.gca()` not `plt.show()`

## Library Modules (`apps/lib/`)

Shared code for regression demos:

| Module | Description |
|--------|-------------|
| `models.py` | `MyBayesianRidge`, `ConformalPrediction`, `NeuralNetworkRegression`, `QuantileRegressionUQ` |
| `kernels.py` | GP kernels (RBF, Matérn, bump, polynomial) |
| `basis.py` | Basis feature functions (RBF, Fourier, LJ) |
| `metrics.py` | Probabilistic metrics (log likelihood, CRPS) |
| `data.py` | Ground truth functions and data generation |
| `optimization.py` | GP hyperparameter optimization |

**Sync workflow:** When modifying lib/, run `--check-sync` to verify inline copies in notebooks match.
