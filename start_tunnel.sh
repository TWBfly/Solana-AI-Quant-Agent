#!/bin/bash
# Solana AI Quant Agent - Enterprise Cloudflare Anycast Tunnel

URL_FILE="/home/ubuntu/Solana-AI-Quant-Agent/public_url.txt"
LOG_FILE="/home/ubuntu/Solana-AI-Quant-Agent/tunnel.log"

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Cloudflare Anycast Tunnel to 127.0.0.1:8001..."
  
  /usr/local/bin/cloudflared tunnel --url http://127.0.0.1:8001 --no-autoupdate 2>&1 | while IFS= read -r line; do
    echo "$line"
    url=$(echo "$line" | grep -oE "https://[a-zA-Z0-9.-]+\.trycloudflare\.com" | head -n 1)
    if [ -n "$url" ]; then
      echo "$url" > "$URL_FILE"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] ★ Cloudflare URL Active: $url"
    fi
  done
  
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cloudflare tunnel disconnected, restarting in 2s..."
  sleep 2
done
