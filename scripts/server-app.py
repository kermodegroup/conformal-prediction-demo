import marimo
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"
WASM_DIR = Path(__file__).parent / "wasm"

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

# Serve WASM assets at /assets/
if WASM_DIR.exists() and (WASM_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(WASM_DIR / "assets")), name="assets")

# Serve WASM extra files at /files/
if WASM_DIR.exists() and (WASM_DIR / "files").exists():
    app.mount("/files", StaticFiles(directory=str(WASM_DIR / "files")), name="files")

@app.get("/", response_class=HTMLResponse)
def index():
    all_notebooks = []
    for name in live_notebooks:
        all_notebooks.append((name, f"/{name}/", "live"))
    for name in wasm_notebooks:
        all_notebooks.append((name, f"/{name}.html", "wasm"))
    all_notebooks.sort(key=lambda x: x[0])

    notebook_links = "".join(
        f'<li><a href="{url}">{name.replace("-", " ").replace("_", " ").title()}</a>'
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
