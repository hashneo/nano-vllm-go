.PHONY: all build test clean run example fmt vet build-purego build-pytorch run-purego run-pytorch

# Binary name
BINARY_NAME=nano-vllm-go
EXAMPLE_BINARY=example

# Build directory
BUILD_DIR=bin

# LibTorch path (adjust as needed)
LIBTORCH_PATH=third_party/libtorch

all: fmt vet test build

# Build the example (default)
build:
	@echo "Building example..."
	@mkdir -p $(BUILD_DIR)
	@go build -o $(BUILD_DIR)/$(EXAMPLE_BINARY) ./example

# Build Pure Go version
build-purego:
	@echo "Building Pure Go version..."
	@mkdir -p $(BUILD_DIR)
	@go build -o $(BUILD_DIR)/simple_example ./purego/example_simple

# Build PyTorch version
build-pytorch:
	@echo "Building PyTorch version..."
	@mkdir -p $(BUILD_DIR)
	@echo "Note: Requires LibTorch installed"
	@go build -tags pytorch -o $(BUILD_DIR)/pytorch_example ./pytorch/example || \
		echo "PyTorch build failed - see pytorch/README.md for setup"

# Build all versions
build-all: build build-purego
	@echo "Building all versions (skipping PyTorch - requires setup)"

# Run the example
run: build
	@echo "Running example..."
	@./$(BUILD_DIR)/$(EXAMPLE_BINARY)

# Run Pure Go example
run-purego: build-purego
	@echo "Running Pure Go example..."
	@./$(BUILD_DIR)/simple_example

# Run PyTorch example
run-pytorch: build-pytorch
	@echo "Running PyTorch example..."
	@export LD_LIBRARY_PATH=$(LIBTORCH_PATH)/lib:$$LD_LIBRARY_PATH && ./$(BUILD_DIR)/pytorch_example

# Run tests
test:
	@echo "Running tests..."
	@go test -v ./...

# Run tests with coverage
coverage:
	@echo "Running tests with coverage..."
	@go test -coverprofile=coverage.out ./...
	@go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

# Format code
fmt:
	@echo "Formatting code..."
	@go fmt ./...

# Run go vet
vet:
	@echo "Running go vet..."
	@go vet ./...

# Run linter (requires golangci-lint)
lint:
	@echo "Running linter..."
	@golangci-lint run

# Tidy dependencies
tidy:
	@echo "Tidying dependencies..."
	@go mod tidy

# Download dependencies
deps:
	@echo "Downloading dependencies..."
	@go mod download

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -rf $(BUILD_DIR)
	@rm -f coverage.out coverage.html

# Install dependencies and build
install: deps tidy build

# Help
help:
	@echo "Available targets:"
	@echo "  all       - Format, vet, test, and build"
	@echo "  build     - Build the example binary"
	@echo "  run       - Build and run the example"
	@echo "  test      - Run tests"
	@echo "  coverage  - Run tests with coverage"
	@echo "  fmt       - Format code"
	@echo "  vet       - Run go vet"
	@echo "  lint      - Run linter (requires golangci-lint)"
	@echo "  tidy      - Tidy dependencies"
	@echo "  deps      - Download dependencies"
	@echo "  clean     - Clean build artifacts"
	@echo "  install   - Download deps, tidy, and build"
	@echo "  help      - Show this help message"
