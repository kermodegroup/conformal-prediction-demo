#!/bin/bash
set -e

# Configuration
DEMOS_DIR=$(realpath $(dirname $0)/..)
REMOTE="sciml"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== SciML Deployment ===${NC}"

# Change to demos directory
cd "$DEMOS_DIR"
echo "Working directory: $(pwd)"

# Categorize notebooks by WASM compatibility
echo -e "\n${YELLOW}Categorizing notebooks...${NC}"
uv run python scripts/categorize_notebooks.py --output-wasm wasm_notebooks.txt --output-live live_notebooks.txt

# Check what we're deploying
echo -e "\n${YELLOW}Deployment summary:${NC}"
WASM_COUNT=$(cat wasm_notebooks.txt 2>/dev/null | grep -c . || echo 0)
LIVE_COUNT=$(cat live_notebooks.txt 2>/dev/null | grep -c . || echo 0)
echo "  WASM notebooks: $WASM_COUNT"
echo "  Live notebooks: $LIVE_COUNT"

# Build WASM notebooks
echo -e "\n${YELLOW}Building WASM notebooks...${NC}"
if [ "$WASM_COUNT" -gt 0 ]; then
    # Clean and recreate output directory
    rm -rf _wasm_site
    mkdir -p _wasm_site

    # Export each WASM-compatible notebook (flat structure - just basename)
    # Note: || [ -n "$notebook" ] handles files without trailing newline
    while IFS= read -r notebook || [ -n "$notebook" ]; do
        [ -z "$notebook" ] && continue
        # Use just the basename for flat URL structure
        basename=$(basename "$notebook" .py)
        echo "  Exporting: $notebook -> ${basename}.html"
        uv run marimo export html-wasm --mode run --no-show-code "$notebook" -o "_wasm_site/${basename}.html"
    done < wasm_notebooks.txt
else
    echo "  No WASM notebooks to build"
fi

# Deploy WASM notebooks
echo -e "\n${YELLOW}Deploying WASM notebooks...${NC}"
if [ -d "_wasm_site" ] && [ "$WASM_COUNT" -gt 0 ]; then
    rsync -avz --delete _wasm_site/ ${REMOTE}:~/marimo-server/wasm/
else
    echo "  No WASM notebooks to deploy"
fi

# Deploy live notebooks (JAX-dependent) - flatten to just basenames
echo -e "\n${YELLOW}Deploying live notebooks...${NC}"
if [ -s live_notebooks.txt ]; then
    # Clear existing notebooks first
    ssh ${REMOTE} 'rm -f ~/marimo-server/notebooks/*.py'
    # Copy each notebook with flat structure
    while IFS= read -r notebook || [ -n "$notebook" ]; do
        [ -z "$notebook" ] && continue
        echo "  Deploying: $notebook"
        scp "$notebook" ${REMOTE}:~/marimo-server/notebooks/
    done < live_notebooks.txt
else
    echo "  No live notebooks to deploy"
    # Clear remote notebooks directory if no live notebooks
    ssh ${REMOTE} 'rm -f ~/marimo-server/notebooks/*.py'
fi

# Restart server
echo -e "\n${YELLOW}Restarting server...${NC}"
ssh ${REMOTE} '~/marimo-server/deploy.sh'

# Cleanup temporary files
rm -f wasm_notebooks.txt live_notebooks.txt

echo -e "\n${GREEN}=== Deployment complete ===${NC}"
echo "Visit: https://sciml.warwick.ac.uk/"
