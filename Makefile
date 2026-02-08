.PHONY: all build test clean run example fmt vet

# Binary name
BINARY_NAME=nano-vllm-go
EXAMPLE_BINARY=example

# Build directory
BUILD_DIR=bin

all: fmt vet test build

# Build the example
build:
	@echo "Building example..."
	@mkdir -p $(BUILD_DIR)
	@go build -o $(BUILD_DIR)/$(EXAMPLE_BINARY) ./example

# Run the example
run: build
	@echo "Running example..."
	@./$(BUILD_DIR)/$(EXAMPLE_BINARY)

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
