#!/usr/bin/env bash
# start.sh — Start the DeText backend and frontend together
# Usage: ./start.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$REPO_ROOT/backend"
FRONTEND_DIR="$REPO_ROOT/frontend"

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
YELLOW='\033[1;33m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${CYAN}${BOLD}[DeText]${RESET} $*"; }
ok()   { echo -e "${GREEN}${BOLD}[  OK  ]${RESET} $*"; }
warn() { echo -e "${YELLOW}${BOLD}[ WARN ]${RESET} $*"; }
err()  { echo -e "${RED}${BOLD}[ ERR  ]${RESET} $*"; }

echo ""
echo -e "${BOLD}╔══════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║       DeText — Startup Script        ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════╝${RESET}"
echo ""

# ── 1. Python check ─────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    err "python3 not found. Please install Python 3.9+."
    exit 1
fi
ok "Python: $(python3 --version)"

# ── 2. Node / npm check ──────────────────────────────────────────────────────
if ! command -v npm &>/dev/null; then
    err "npm not found. Please install Node.js (https://nodejs.org)."
    exit 1
fi
ok "Node:   $(node --version)  /  npm $(npm --version)"

# ── 3. uvicorn check ─────────────────────────────────────────────────────────
if ! python3 -m uvicorn --version &>/dev/null; then
    warn "uvicorn not found — installing..."
    pip3 install uvicorn fastapi 2>/dev/null
fi
ok "uvicorn ready"

# ── 4. frontend/node_modules ─────────────────────────────────────────────────
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    log "node_modules not found — running npm install..."
    npm install --prefix "$FRONTEND_DIR"
    ok "npm install complete"
else
    ok "node_modules already present — skipping npm install"
fi

# ── 5. Backend weights (auto-download via main.py if any are missing) ────────
WEIGHTS_DIR="$BACKEND_DIR/weights"
REQUIRED_FILES=(
    "label_encoder.pkl"
    "vectorizer_char_wb_2_4.pkl"
    "vectorizer_char_wb_1_3_langdetect.pkl"
    "clf_ComplementNB.pkl"
    "clf_PassiveAggressive.pkl"
    "clf_RidgeClassifier.pkl"
    "clf_SGDClassifier.pkl"
    "langdetect_style_complement_nb.pkl"
    "fasttext_weights.pth"
    "glotlid_weights.pth"
    "cld3_weights.pth"
    "charcnn_highcap_weights.pth"
)
MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    [ ! -f "$WEIGHTS_DIR/$f" ] && MISSING=$((MISSING + 1))
done

if [ "$MISSING" -gt 0 ]; then
    warn "$MISSING weight file(s) missing."
    log "Running download_weights.py — this may take a while (~2.9 GB total)..."
    python3 "$REPO_ROOT/download_weights.py"
else
    ok "All weight files present"
fi

# ── 6. Launch backend and frontend in parallel ────────────────────────────────
echo ""
log "Starting backend  →  http://localhost:8000"
log "Starting frontend →  http://localhost:5173"
echo ""

cleanup() {
    echo ""
    log "Shutting down..."
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
    wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
    ok "Stopped. Goodbye."
}
trap cleanup SIGINT SIGTERM

# Backend
cd "$BACKEND_DIR"
python3 -W ignore -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Frontend
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!

cd "$REPO_ROOT"

ok "Both services running. Press Ctrl+C to stop."
echo ""

wait "$BACKEND_PID" "$FRONTEND_PID"
