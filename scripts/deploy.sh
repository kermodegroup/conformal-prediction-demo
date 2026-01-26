#!/bin/bash
set -e

NOTEBOOKS_DIR=/home/ubuntu/marimo-server/notebooks
WASM_DIR=/var/www/marimo-wasm
BASE_PORT=2718
DOMAIN=sciml.jrkermode.uk

# Find live notebooks (need marimo server)
live_notebooks=()
if [ -d "$NOTEBOOKS_DIR" ]; then
    while IFS= read -r -d '' f; do
        live_notebooks+=("$(basename "$f" .py)")
    done < <(find "$NOTEBOOKS_DIR" -maxdepth 1 -name "*.py" -print0 2>/dev/null)
fi

# Find WASM notebooks (static HTML in apps/ subdirectory)
wasm_notebooks=()
if [ -d "$WASM_DIR/apps" ]; then
    while IFS= read -r -d '' f; do
        wasm_notebooks+=("$(basename "$f" .html)")
    done < <(find "$WASM_DIR/apps" -maxdepth 1 -name "*.html" -print0 2>/dev/null)
fi

echo "Live notebooks: ${live_notebooks[*]:-none}"
echo "WASM notebooks: ${wasm_notebooks[*]:-none}"

# Clean up old services
echo "Cleaning up old services..."
for service in /etc/systemd/system/marimo-*.service; do
    [ -e "$service" ] || continue
    service_name=$(basename "$service" .service)
    notebook_name=${service_name#marimo-}
    
    if [[ ! " ${live_notebooks[*]} " =~ " ${notebook_name} " ]]; then
        echo "Removing old service: ${service_name}"
        sudo systemctl stop "$service_name" 2>/dev/null || true
        sudo systemctl disable "$service_name" 2>/dev/null || true
        sudo rm "$service"
    fi
done

# Generate systemd services for live notebooks
for i in "${!live_notebooks[@]}"; do
    name="${live_notebooks[$i]}"
    port=$((BASE_PORT + i))
    
    sudo tee /etc/systemd/system/marimo-${name}.service > /dev/null << SYSTEMD
[Unit]
Description=Marimo - ${name}
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/marimo-server
Environment="PATH=/home/ubuntu/marimo-server/.venv/bin:/home/ubuntu/.local/bin:/usr/bin"
ExecStart=/home/ubuntu/marimo-server/.venv/bin/marimo run --headless --host 127.0.0.1 --port ${port} --sandbox /home/ubuntu/marimo-server/notebooks/${name}.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SYSTEMD
    echo "Created service marimo-${name} on port ${port}"
done

# Generate nginx config
{
    # Upstreams for live notebooks
    for i in "${!live_notebooks[@]}"; do
        name="${live_notebooks[$i]}"
        port=$((BASE_PORT + i))
        echo "upstream marimo-${name} { server 127.0.0.1:${port}; }"
    done
    
    cat << NGINX

server {
    listen 80;
    server_name ${DOMAIN};
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl;
    server_name ${DOMAIN};

    ssl_certificate /etc/letsencrypt/live/${DOMAIN}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${DOMAIN}/privkey.pem;

    auth_basic "Marimo Notebooks";
    auth_basic_user_file /etc/nginx/.htpasswd;

    # Serve WASM static files (no auth - JS fetch() doesn't send credentials)
    location /wasm/ {
        auth_basic off;
        alias /var/www/marimo-wasm/;
        try_files \$uri \$uri/ =404;

        # CORS headers required for WASM/Pyodide to fetch packages
        add_header Access-Control-Allow-Origin "*" always;
        add_header Access-Control-Allow-Methods "GET, OPTIONS" always;
        add_header Access-Control-Allow-Headers "*" always;
    }

    # Index page
    location = / {
        default_type text/html;
        return 200 '<!DOCTYPE html>
<html>
<head><title>SciML Notebooks</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
h1 { color: #333; }
h2 { color: #666; margin-top: 2em; }
ul { list-style: none; padding: 0; }
li { margin: 10px 0; }
a { color: #0066cc; text-decoration: none; font-size: 1.1em; }
a:hover { text-decoration: underline; }
.badge { font-size: 0.7em; padding: 2px 6px; border-radius: 3px; margin-left: 8px; }
.wasm { background: #d4edda; color: #155724; }
.live { background: #fff3cd; color: #856404; }
</style>
</head>
<body>
<h1>SciML Notebooks</h1>
NGINX

    # WASM notebooks section
    if [ ${#wasm_notebooks[@]} -gt 0 ]; then
        echo '<h2>Static Notebooks (WASM)</h2>'
        echo '<ul>'
        for name in "${wasm_notebooks[@]}"; do
            title=$(echo "$name" | sed 's/[-_]/ /g' | sed 's/\b./\u&/g')
            echo "<li><a href=\"/wasm/apps/${name}.html\">${title}</a><span class=\"badge wasm\">WASM</span></li>"
        done
        echo '</ul>'
    fi

    # Live notebooks section
    if [ ${#live_notebooks[@]} -gt 0 ]; then
        echo '<h2>Interactive Notebooks (Live Server)</h2>'
        echo '<ul>'
        for name in "${live_notebooks[@]}"; do
            title=$(echo "$name" | sed 's/[-_]/ /g' | sed 's/\b./\u&/g')
            echo "<li><a href=\"/${name}/\">${title}</a><span class=\"badge live\">Live</span></li>"
        done
        echo '</ul>'
    fi

    cat << 'NGINX'
</body>
</html>';
    }
NGINX

    # Location blocks for live notebooks
    for name in "${live_notebooks[@]}"; do
        cat << NGINX

    location /${name}/ {
        proxy_pass http://marimo-${name}/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 600;
    }
NGINX
    done

    echo "}"
} | sudo tee /etc/nginx/sites-available/marimo > /dev/null

# Reload services
sudo systemctl daemon-reload

for name in "${live_notebooks[@]}"; do
    sudo systemctl enable "marimo-${name}"
    sudo systemctl restart "marimo-${name}"
done

sudo nginx -t && sudo systemctl reload nginx

echo ""
echo "Done! Deployment complete:"
echo "  WASM notebooks: ${#wasm_notebooks[@]}"
echo "  Live notebooks: ${#live_notebooks[@]}"
echo "  URL: https://${DOMAIN}/"
