#!/usr/bin/env python3
"""Generate index.html for GitHub Pages WASM deployment."""

import argparse
import tomllib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="_site")
    args = parser.parse_args()

    # Load demo config for titles/ordering
    config_file = Path("demos.toml")
    demo_config = []
    if config_file.exists():
        with open(config_file, "rb") as f:
            config = tomllib.load(f)
            demo_config = config.get("demos", [])

    config_by_name = {d["name"]: d for d in demo_config}

    # Find built WASM files
    output_dir = Path(args.output_dir)
    wasm_files = sorted(output_dir.glob("*.html"))
    wasm_files = [f for f in wasm_files if f.name != "index.html"]

    # Build notebook list
    notebooks = []
    for f in wasm_files:
        name = f.stem
        cfg = config_by_name.get(name, {})
        # Skip hidden notebooks
        if cfg.get("hidden", False):
            continue
        title = cfg.get("title", name.replace("-", " ").title())
        notebooks.append((name, title))

    # Sort by config order
    config_order = [d["name"] for d in demo_config]
    notebooks.sort(
        key=lambda x: (config_order.index(x[0]) if x[0] in config_order else 999, x[0])
    )

    # Generate HTML
    links = "\n".join(
        f'        <li><a href="{name}.html">{title}</a> <span class="badge wasm">WASM</span></li>'
        for name, title in notebooks
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SciML Demos - University of Warwick</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
        h1 {{ color: #5f259f; }}
        ul {{ list-style: none; padding: 0; }}
        li {{ margin: 10px 0; }}
        a {{ color: #0066cc; text-decoration: none; font-size: 1.1em; }}
        a:hover {{ text-decoration: underline; }}
        .badge {{ font-size: 0.7em; padding: 2px 6px; border-radius: 3px; margin-left: 8px; }}
        .wasm {{ background: #d4edda; color: #155724; }}
        .live {{ background: #fff3cd; color: #856404; }}
        .note {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 1em; margin: 1.5em 0; }}
    </style>
</head>
<body>
    <h1>SciML Notebooks</h1>
    <p>Interactive scientific machine learning demonstrations.
    Developed by <a href="https://warwick.ac.uk/jrkermode">James Kermode</a>.</p>

    <h2>Interactive Demos (WASM)</h2>
    <p>These run entirely in your browser - no server required.</p>
    <ul>
{links}
    </ul>

    <div class="note">
        <h3>Live Notebooks</h3>
        <p>Demos requiring JAX or other native dependencies are available at
        <a href="https://sciml.warwick.ac.uk/">sciml.warwick.ac.uk</a>
        (University of Warwick SSO required).</p>
    </div>
</body>
</html>
"""

    (output_dir / "index.html").write_text(html)
    print(f"Generated {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
