#!/bin/bash
# Solana AI Quant Agent - Autonomous Public Tunnel with Active Health Watchdog

KEY="/home/ubuntu/.ssh/id_solana"
URL_FILE="/home/ubuntu/Solana-AI-Quant-Agent/public_url.txt"

# Ensure dedicated SSH key exists
if [ ! -f "$KEY" ]; then
  ssh-keygen -t ed25519 -f "$KEY" -N "" -q
fi

# Active Watchdog & Keep-Alive Daemon
# 1. Pings the tunnel endpoint every 25 seconds so localhost.run never times out on inactivity
# 2. If the tunnel drops or returns non-200, forcefully terminates the zombie SSH session to auto-reconnect
(
  while true; do
    sleep 25
    if [ -f "$URL_FILE" ]; then
      CURRENT_URL=$(head -n 1 "$URL_FILE" 2>/dev/null | tr -d '[:space:]')
      if [ -n "$CURRENT_URL" ]; then
        STATUS=$(curl -s -m 6 -o /dev/null -w "%{http_code}" "$CURRENT_URL/api/tokens" 2>/dev/null)
        if [ "$STATUS" != "200" ]; then
          echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Watchdog] Tunnel $CURRENT_URL health check failed (HTTP $STATUS), restarting tunnel..."
          pkill -f "id_solana.*nokey@localhost.run" 2>/dev/null
        fi
      fi
    fi
  done
) &
WATCHDOG_PID=$!

trap "kill $WATCHDOG_PID 2>/dev/null" EXIT

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting tunnel with dedicated key $KEY..."

while true; do
  ssh -i "$KEY" \
      -o StrictHostKeyChecking=no \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=2 \
      -o ExitOnForwardFailure=yes \
      -R 80:localhost:8001 \
      nokey@localhost.run 2>&1 | while IFS= read -r line; do
        echo "$line"
        url=$(echo "$line" | grep -oE "https://[a-zA-Z0-9.-]+\.lhr\.life" | head -n 1)
        if [ -n "$url" ]; then
          echo "$url" > "$URL_FILE"
          echo "[$(date '+%Y-%m-%d %H:%M:%S')] Public URL active: $url"
        fi
      done
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] SSH tunnel exited, reconnecting in 2s..."
  sleep 2
done
