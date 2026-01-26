#!/usr/bin/env python3
"""
Categorize notebooks into WASM-compatible and live-server-required.
"""

import argparse
import re
from pathlib import Path

WASM_INCOMPATIBLE_DEPS = {
    'jax',
    'jaxlib',
    'tensorflow',
    'torch',
    'pytorch',
    'scikit-learn',
    'sklearn',
}


def check_wasm_compatible(notebook_path: Path) -> tuple[bool, list[str]]:
    """Check if a notebook's dependencies are WASM-compatible."""
    content = notebook_path.read_text()
    
    metadata_match = re.search(r'# /// script\n(.*?)# ///', content, re.DOTALL)
    
    incompatible_found = []
    
    if metadata_match:
        metadata = metadata_match.group(1).lower()
        for dep in WASM_INCOMPATIBLE_DEPS:
            if dep in metadata:
                incompatible_found.append(dep)
    
    for dep in WASM_INCOMPATIBLE_DEPS:
        if re.search(rf'^\s*(import|from)\s+{dep}\b', content, re.MULTILINE):
            if dep not in incompatible_found:
                incompatible_found.append(dep)
    
    return (len(incompatible_found) == 0, incompatible_found)


def main():
    parser = argparse.ArgumentParser(description="Categorize notebooks by WASM compatibility")
    parser.add_argument("--output-wasm", default="wasm_notebooks.txt")
    parser.add_argument("--output-live", default="live_notebooks.txt")
    args = parser.parse_args()

    wasm_notebooks = []
    live_notebooks = []

    for directory in ["notebooks", "apps"]:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue

        for path in dir_path.rglob("*.py"):
            if "lib" not in path.parts:
                is_compatible, incompatible_deps = check_wasm_compatible(path)
                
                if is_compatible:
                    wasm_notebooks.append(str(path))
                    print(f"WASM: {path}")
                else:
                    live_notebooks.append(str(path))
                    print(f"LIVE: {path} ({', '.join(incompatible_deps)})")

    with open(args.output_wasm, "w") as f:
        f.write("\n".join(wasm_notebooks))
    
    with open(args.output_live, "w") as f:
        f.write("\n".join(live_notebooks))

    print(f"\nSummary: {len(wasm_notebooks)} WASM, {len(live_notebooks)} live")


if __name__ == "__main__":
    main()
