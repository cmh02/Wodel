#!/usr/bin/env bash

# Graceful cleanup function to shut down both backend and frontend servers
cleanup() {
    echo ""
    echo "🛑 Shutting down Wodel services..."
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}

trap cleanup INT TERM

echo "========================================="
echo "    Starting Wodel Development Stack"
echo "========================================="

echo "1/2 Starting Backend (FastAPI on http://localhost:8000)..."
uv run uvicorn engine.server:app --reload --port 8000 &

echo "2/2 Starting Frontend (Vite React on http://localhost:5173)..."
(cd frontend && npm run dev) &

echo "========================================="
echo "  Press Ctrl+C to stop both servers"
echo "========================================="

wait
