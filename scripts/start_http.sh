#!/bin/bash
# Quick start script for HTTP Backend Mode (Python server)

set -e

echo "========================================="
echo "Nano-vLLM-Go - HTTP Backend Mode"
echo "========================================="
echo ""

# Default model
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-0.5B-Instruct}"
SERVER_PORT="${SERVER_PORT:-8000}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    echo "Please install Python 3.8+"
    exit 1
fi

# Check/install dependencies
echo "Checking Python dependencies..."
if ! python3 -c "import flask, torch, transformers" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip3 install flask torch transformers || {
        echo "❌ Failed to install dependencies"
        exit 1
    }
fi
echo "✓ Python dependencies ready"
echo ""

# Build Go client
echo "Building Go client..."
go build -o bin/http_test ./purego/example_http || {
    echo "❌ Build failed"
    exit 1
}
echo "✓ Build complete"
echo ""

# Get prompt from args or use default
if [ $# -eq 0 ]; then
    PROMPT="What is the capital of France?"
else
    PROMPT="$*"
fi

# Check if server is already running
if curl -s http://localhost:$SERVER_PORT/health > /dev/null 2>&1; then
    echo "✓ Server already running on port $SERVER_PORT"
    echo ""
else
    # Start server in background
    echo "Starting Python inference server..."
    echo "Model: $MODEL_NAME"
    echo "Port: $SERVER_PORT"
    echo ""

    python3 server.py --model "$MODEL_NAME" --port $SERVER_PORT > /tmp/nano-vllm-server.log 2>&1 &
    SERVER_PID=$!

    # Save PID
    echo $SERVER_PID > /tmp/nano-vllm-server.pid

    echo "Server PID: $SERVER_PID"
    echo "Log: /tmp/nano-vllm-server.log"

    # Wait for server to be ready
    echo ""
    echo "Waiting for server to load model..."
    for i in {1..60}; do
        if curl -s http://localhost:$SERVER_PORT/health > /dev/null 2>&1; then
            echo "✓ Server ready!"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "❌ Server failed to start. Check log:"
            echo "   tail /tmp/nano-vllm-server.log"
            exit 1
        fi
        echo -n "."
        sleep 1
    done
    echo ""
fi

# Run client
echo ""
echo "========================================="
echo "Running HTTP Client"
echo "========================================="
echo ""

export MODEL_SERVER="http://localhost:$SERVER_PORT"

./bin/http_test "$PROMPT"

echo ""
echo "========================================="
echo "💡 Tips:"
echo "  • HTTP = Fastest setup (5 min)"
echo "  • Speed: ~40-80 tok/s (depends on model)"
echo "  • Perfect for development & testing"
echo ""
echo "Try another prompt:"
echo "  $0 \"Your question here\""
echo ""
echo "Stop server:"
echo "  kill \$(cat /tmp/nano-vllm-server.pid)"
echo ""
echo "Check logs:"
echo "  tail -f /tmp/nano-vllm-server.log"
echo ""
echo "Use different model:"
echo "  MODEL_NAME='TinyLlama/TinyLlama-1.1B-Chat-v1.0' $0"
echo "========================================="
