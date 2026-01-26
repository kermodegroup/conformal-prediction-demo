#!/usr/bin/env python3
"""
Build script for marimo notebooks.

Handles:
- Syncing lib/ modules into marimo files for WASM export
- Exporting notebooks to HTML-WASM format
- Generating index.html
- Skipping notebooks with WASM-incompatible dependencies
"""

import os
import re
import subprocess
import argparse
import tempfile
import shutil
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# Dependencies that don't work in WASM
WASM_INCOMPATIBLE_DEPS = {
    'jax',
    'jaxlib', 
    'tensorflow',
    'torch',
    'pytorch',
    'scikit-learn',
    'sklearn',
}


def check_wasm_compatible(notebook_path: Path) -> Tuple[bool, List[str]]:
    """
    Check if a notebook's dependencies are WASM-compatible.
    
    Returns (is_compatible, list_of_incompatible_deps)
    """
    content = notebook_path.read_text()
    
    # Look for PEP 723 inline metadata
    metadata_match = re.search(r'# /// script\n(.*?)# ///', content, re.DOTALL)
    
    incompatible_found = []
    
    if metadata_match:
        metadata = metadata_match.group(1).lower()
        for dep in WASM_INCOMPATIBLE_DEPS:
            if dep in metadata:
                incompatible_found.append(dep)
    
    # Also check imports in the code itself
    for dep in WASM_INCOMPATIBLE_DEPS:
        # Check for import statements
        if re.search(rf'^\s*(import|from)\s+{dep}\b', content, re.MULTILINE):
            if dep not in incompatible_found:
                incompatible_found.append(dep)
    
    return (len(incompatible_found) == 0, incompatible_found)


# ... [keep all the existing functions unchanged until main()] ...


def main() -> None:
    parser = argparse.ArgumentParser(description="Build marimo notebooks")
    parser.add_argument(
        "--output-dir", default="_site", help="Output directory for built files"
    )
    parser.add_argument(
        "--sync-lib", action="store_true",
        help="Bundle lib/ modules with marimo files for WASM export"
    )
    parser.add_argument(
        "--check-sync", action="store_true",
        help="Check if lib/ modules are in sync with inline code (no build)"
    )
    parser.add_argument(
        "--skip-incompatible", action="store_true",
        help="Skip notebooks with WASM-incompatible dependencies instead of failing"
    )
    args = parser.parse_args()

    # If just checking sync, do that and exit
    if args.check_sync:
        apps_dir = Path("apps")
        if apps_dir.exists():
            generate_sync_report(apps_dir)
        else:
            print("No apps/ directory found")
        return

    all_notebooks: List[str] = []
    skipped_notebooks: List[Tuple[str, List[str]]] = []
    temp_files: List[Path] = []  # Track temp files for cleanup

    for directory in ["notebooks", "apps"]:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: Directory not found: {dir_path}")
            continue

        # Collect notebooks, excluding lib/ subdirectory
        for path in dir_path.rglob("*.py"):
            if "lib" not in path.parts:
                # Check WASM compatibility
                is_compatible, incompatible_deps = check_wasm_compatible(path)
                
                if not is_compatible:
                    if args.skip_incompatible:
                        print(f"Skipping {path} - incompatible deps: {', '.join(incompatible_deps)}")
                        skipped_notebooks.append((str(path), incompatible_deps))
                        continue
                    else:
                        print(f"Warning: {path} has WASM-incompatible deps: {', '.join(incompatible_deps)}")
                
                all_notebooks.append(str(path))

    if not all_notebooks:
        print("No compatible notebooks found!")
        return

    try:
        successful_notebooks: List[str] = []
        
        # Export notebooks sequentially
        for nb in all_notebooks:
            nb_path = Path(nb)
            lib_dir = nb_path.parent / "lib"

            # Bundle with lib/ if it exists and sync is enabled
            if args.sync_lib and lib_dir.exists():
                print(f"Bundling {nb} with lib/ modules...")
                bundled_path = bundle_marimo_with_lib(nb_path, lib_dir)
                temp_files.append(bundled_path.parent)  # Track temp dir for cleanup
                export_path = str(bundled_path)
                # Use original path for output filename
                output_name = nb
            else:
                export_path = nb
                output_name = None

            success = export_html_wasm(
                export_path,
                args.output_dir,
                as_app=nb.startswith("apps/"),
                output_name=output_name
            )
            
            if success:
                successful_notebooks.append(nb)
            elif args.skip_incompatible:
                print(f"  Export failed, skipping {nb}")
                skipped_notebooks.append((nb, ["export_failed"]))

        # Generate index only for successfully exported notebooks
        generate_index(successful_notebooks, args.output_dir)
        
        # Summary
        if skipped_notebooks:
            print(f"\n=== Build Summary ===")
            print(f"Exported: {len(successful_notebooks)} notebooks")
            print(f"Skipped: {len(skipped_notebooks)} notebooks")
            for nb, reasons in skipped_notebooks:
                print(f"  - {nb}: {', '.join(reasons)}")

    finally:
        # Cleanup temp files
        for temp_dir in temp_files:
            if temp_dir.exists() and str(temp_dir).startswith(tempfile.gettempdir()):
                shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
