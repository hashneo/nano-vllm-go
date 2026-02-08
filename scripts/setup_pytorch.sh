#!/bin/bash
# Setup script for PyTorch implementation

set -e

echo "Nano-vLLM-Go PyTorch Setup"
echo "=========================="
echo ""

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "Detected OS: $OS"
echo "Detected Architecture: $ARCH"
echo ""

# Create third_party directory
mkdir -p third_party
cd third_party

# Download LibTorch
if [ "$OS" = "Linux" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        echo "Downloading LibTorch for Linux (CPU)..."
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
        LIBTORCH_FILE="libtorch-linux-cpu.zip"

        if [ ! -f "$LIBTORCH_FILE" ]; then
            wget -O "$LIBTORCH_FILE" "$LIBTORCH_URL"
        fi

        if [ ! -d "libtorch" ]; then
            unzip -q "$LIBTORCH_FILE"
        fi

        echo "LibTorch installed to: $(pwd)/libtorch"
    fi
elif [ "$OS" = "Darwin" ]; then
    if [ "$ARCH" = "arm64" ]; then
        echo "Downloading LibTorch for macOS (Apple Silicon)..."
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.1.0.zip"
    else
        echo "Downloading LibTorch for macOS (Intel)..."
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip"
    fi

    LIBTORCH_FILE="libtorch-macos.zip"

    if [ ! -f "$LIBTORCH_FILE" ]; then
        curl -L -o "$LIBTORCH_FILE" "$LIBTORCH_URL"
    fi

    if [ ! -d "libtorch" ]; then
        unzip -q "$LIBTORCH_FILE"
    fi

    echo "LibTorch installed to: $(pwd)/libtorch"
else
    echo "Unsupported OS: $OS"
    exit 1
fi

cd ..

# Build C++ wrapper
echo ""
echo "Building PyTorch C++ wrapper..."
cd pytorch

if [ "$OS" = "Linux" ]; then
    g++ -shared -fPIC -o libpytorch_wrapper.so \
        model_runner_wrapper.cpp \
        -I../third_party/libtorch/include \
        -I../third_party/libtorch/include/torch/csrc/api/include \
        -L../third_party/libtorch/lib \
        -ltorch -ltorch_cpu -lc10 \
        -std=c++17 \
        -Wl,-rpath,../third_party/libtorch/lib
elif [ "$OS" = "Darwin" ]; then
    clang++ -shared -fPIC -o libpytorch_wrapper.dylib \
        model_runner_wrapper.cpp \
        -I../third_party/libtorch/include \
        -I../third_party/libtorch/include/torch/csrc/api/include \
        -L../third_party/libtorch/lib \
        -ltorch -ltorch_cpu -lc10 \
        -std=c++17 \
        -Wl,-rpath,@loader_path/../third_party/libtorch/lib
fi

cd ..

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set library path:"
if [ "$OS" = "Linux" ]; then
    echo "   export LD_LIBRARY_PATH=\$(pwd)/third_party/libtorch/lib:\$LD_LIBRARY_PATH"
elif [ "$OS" = "Darwin" ]; then
    echo "   export DYLD_LIBRARY_PATH=\$(pwd)/third_party/libtorch/lib:\$DYLD_LIBRARY_PATH"
fi
echo ""
echo "2. Build PyTorch version:"
echo "   go build -tags pytorch -o bin/pytorch_example ./pytorch/example"
echo ""
echo "3. Export model to TorchScript:"
echo "   python3 scripts/export_model.py"
echo ""
