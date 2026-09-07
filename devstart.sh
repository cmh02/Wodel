#!/usr/bin/env bash

# Graceful cleanup function to shut down backend and frontend on any port holding them
cleanup() {
    echo ""
    echo "🛑 Shutting down Wodel services..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:5173 | xargs kill -9 2>/dev/null || true
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}

trap cleanup INT TERM

# Ensure ports 8000 and 5173 are free before starting new instances
echo "Checking and freeing ports 8000 and 5173..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true

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
