# SciML Notebooks

Interactive demonstrations of our research using [marimo](https://marimo.io) notebooks.

**Live site:** https://sciml.warwick.ac.uk

## Deployment Modes

Notebooks are automatically deployed in one of two modes based on their dependencies:

### WASM Notebooks (Static)
- Run entirely in the browser using WebAssembly/Pyodide
- No server-side computation required
- Served as static HTML files
- Limited to pure Python packages available in Pyodide

### Live Notebooks (Server)
- Run on the server with full Python environment
- Required for notebooks with native dependencies (JAX, PyTorch, watchdog, etc.)
- Each notebook runs as a separate marimo process
- Proxied through nginx with WebSocket support

## Adding Notebooks

1. Add your notebook to `apps/` (for WASM-compatible) or `notebooks/` (for live)
2. Include dependencies in PEP 723 script metadata:
   ```python
   # /// script
   # requires-python = ">=3.12"
   # dependencies = [
   #     "marimo",
   #     "numpy",
   #     "matplotlib",
   # ]
   # ///
   ```
3. Push to `main` - CI/CD will automatically categorize and deploy

## Automatic Categorization

The `scripts/categorize_notebooks.py` script determines deployment mode by checking for WASM-incompatible dependencies:

- **Native extensions:** jax, torch, tensorflow, watchdog, psutil, opencv-python
- **Database drivers:** psycopg2, mysqlclient
- **Crypto libraries:** cryptography, bcrypt
- **And others** (see script for full list)

Notebooks with any incompatible dependency are deployed as live notebooks.

## Local Development

```bash
# Run a notebook locally
marimo edit apps/my-notebook.py

# Test WASM compatibility check
python scripts/categorize_notebooks.py --offline

# Export to WASM HTML (for testing)
marimo export html-wasm apps/my-notebook.py -o output.html
```

## Infrastructure

- **Server:** Deployed via `scripts/deploy.sh`
- **WASM:** Static files served from `/var/www/marimo-wasm/`
- **Live:** systemd services with nginx reverse proxy
- **Auth:** Basic auth for live notebooks, WASM notebooks are public
