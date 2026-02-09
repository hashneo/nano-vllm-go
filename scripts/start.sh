#!/bin/bash
# Master startup script - choose your mode!

set -e

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           Nano-vLLM-Go - Real Model Inference                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if mode specified
if [ $# -eq 0 ]; then
    echo "Choose your inference mode:"
    echo ""
    echo "  1. 🔥 HTTP Backend    (Fastest setup - 5 min, ~40-80 tok/s)"
    echo "  2. ⚡ ONNX Runtime    (Good speed - 15 min, ~20-40 tok/s)"
    echo "  3. 🎓 Pure Go         (Educational - 5 min, ~5-10 tok/s)"
    echo ""
    echo "Usage:"
    echo "  $0 http   [prompt]  - Start HTTP backend mode"
    echo "  $0 onnx   [prompt]  - Start ONNX runtime mode"
    echo "  $0 native [prompt]  - Start pure Go mode"
    echo ""
    echo "Examples:"
    echo "  $0 http \"What is AI?\""
    echo "  $0 onnx \"Explain quantum computing\""
    echo "  $0 native \"Once upon a time\""
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Comparison:"
    echo ""
    echo "┌─────────────┬──────────────┬─────────────┬─────────────────┐"
    echo "│ Mode        │ Setup Time   │ Speed       │ Best For        │"
    echo "├─────────────┼──────────────┼─────────────┼─────────────────┤"
    echo "│ HTTP        │ 5 min        │ 40-80 tok/s │ Development     │"
    echo "│ ONNX        │ 15 min       │ 20-40 tok/s │ Production      │"
    echo "│ Pure Go     │ 5 min        │ 5-10 tok/s  │ Learning        │"
    echo "└─────────────┴──────────────┴─────────────┴─────────────────┘"
    echo ""
    echo "📚 Documentation:"
    echo "  • HTTP:     TEST_REAL_MODEL.md"
    echo "  • ONNX:     ONNX_GUIDE.md"
    echo "  • Pure Go:  NATIVE_TRANSFORMER_GUIDE.md"
    echo ""
    exit 0
fi

MODE="$1"
shift

# Dispatch to appropriate script
case "$MODE" in
    http|HTTP)
        exec ./scripts/start_http.sh "$@"
        ;;
    onnx|ONNX)
        exec ./scripts/start_onnx.sh "$@"
        ;;
    native|go|purego|NATIVE|GO|PUREGO)
        exec ./scripts/start_native.sh "$@"
        ;;
    *)
        echo "❌ Unknown mode: $MODE"
        echo ""
        echo "Valid modes: http, onnx, native"
        echo ""
        echo "Run without arguments to see usage:"
        echo "  $0"
        exit 1
        ;;
esac
