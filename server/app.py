import marimo
import tomllib
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"
WASM_DIR = Path(__file__).parent / "wasm"
CONFIG_FILE = Path(__file__).parent / "demos.toml"

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create marimo server for live notebooks
server = marimo.create_asgi_app()
live_notebooks = []
for notebook in sorted(NOTEBOOKS_DIR.glob("*.py")):
    name = notebook.stem
    server = server.with_app(path=f"/{name}", root=str(notebook))
    live_notebooks.append(name)

# Find WASM notebooks
wasm_notebooks = []
if WASM_DIR.exists():
    for html_file in sorted(WASM_DIR.glob("*.html")):
        if html_file.name != "index.html":
            wasm_notebooks.append(html_file.stem)

# Load demo config
demo_config = []
if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "rb") as f:
        config = tomllib.load(f)
        demo_config = config.get("demos", [])

# Build config lookup
config_by_name = {d["name"]: d for d in demo_config}
config_order = [d["name"] for d in demo_config]

# Serve WASM assets at /assets/
if WASM_DIR.exists() and (WASM_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(WASM_DIR / "assets")), name="assets")

# Serve WASM extra files at /files/
if WASM_DIR.exists() and (WASM_DIR / "files").exists():
    app.mount("/files", StaticFiles(directory=str(WASM_DIR / "files")), name="files")

def get_display_title(name):
    """Get display title from config or auto-generate."""
    if name in config_by_name:
        return config_by_name[name].get("title", name.replace("-", " ").replace("_", " ").title())
    return name.replace("-", " ").replace("_", " ").title()

def get_sort_key(name):
    """Get sort key - config order first, then alphabetical."""
    if name in config_order:
        return (0, config_order.index(name))
    return (1, name)

@app.get("/", response_class=HTMLResponse)
def index():
    all_notebooks = []
    for name in live_notebooks:
        if name not in config_by_name or not config_by_name[name].get("hidden", False):
            all_notebooks.append((name, f"/{name}/", "live"))
    for name in wasm_notebooks:
        if name not in config_by_name or not config_by_name[name].get("hidden", False):
            all_notebooks.append((name, f"/{name}.html", "wasm"))

    # Sort by config order, then alphabetically
    all_notebooks.sort(key=lambda x: get_sort_key(x[0]))

    notebook_links = "".join(
        f'<li><a href="{url}">{get_display_title(name)}</a>'
        f'<span class="badge {badge_type}">{badge_type.upper()}</span></li>'
        for name, url, badge_type in all_notebooks
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SciML - University of Warwick</title>
        <style>
            body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
            h1 {{ color: #5f259f; }}
            ul {{ list-style: none; padding: 0; }}
            li {{ margin: 10px 0; }}
            a {{ color: #0066cc; text-decoration: none; font-size: 1.1em; }}
            a:hover {{ text-decoration: underline; }}
            .badge {{ font-size: 0.7em; padding: 2px 6px; border-radius: 3px; margin-left: 8px; text-transform: uppercase; }}
            .wasm {{ background: #d4edda; color: #155724; }}
            .live {{ background: #fff3cd; color: #856404; }}
        </style>
    </head>
    <body>
        <h1>SciML Notebooks</h1>
        <p>Interactive scientific machine learning demonstrations.</p>
        <ul>{notebook_links}</ul>
    </body>
    </html>
    """

# Explicit route for each WASM notebook HTML file
for wasm_name in wasm_notebooks:
    wasm_path = WASM_DIR / f"{wasm_name}.html"
    app.add_api_route(
        f"/{wasm_name}.html",
        lambda p=wasm_path: FileResponse(p, media_type="text/html"),
        methods=["GET"],
    )

# Mount marimo server last (catch-all for live notebooks)
app.mount("/", server.build())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=2718)
