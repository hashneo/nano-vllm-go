.PHONY: all build clean test fmt vet tidy help

# Build directory
BIN_DIR=bin

# Main binaries
all: ask-gpt2 ask-llama generic-runner simple-demo

# Build individual binaries
ask-gpt2:
	@echo "Building ask-gpt2..."
	@mkdir -p $(BIN_DIR)
	@go build -o $(BIN_DIR)/ask-gpt2 ./cmd/ask-gpt2

ask-llama:
	@echo "Building ask-llama..."
	@mkdir -p $(BIN_DIR)
	@go build -o $(BIN_DIR)/ask-llama ./cmd/ask-llama

generic-runner:
	@echo "Building generic-runner..."
	@mkdir -p $(BIN_DIR)
	@go build -o $(BIN_DIR)/generic-runner ./cmd/generic-runner

simple-demo:
	@echo "Building simple-demo..."
	@mkdir -p $(BIN_DIR)
	@go build -o $(BIN_DIR)/simple-demo ./cmd/simple-demo

# Convenience target
build: all

# Run main demo
demo: ask-gpt2
	@./demo_capitals.sh

# Run tests
test:
	@echo "Running tests..."
	@go test -v ./...

# Format code
fmt:
	@echo "Formatting code..."
	@go fmt ./...

# Run go vet
vet:
	@echo "Running go vet..."
	@go vet ./...

# Tidy dependencies
tidy:
	@echo "Tidying dependencies..."
	@go mod tidy

# Clean binaries
clean:
	@echo "Cleaning binaries..."
	@rm -rf $(BIN_DIR)
	@echo "Done!"

# Help
help:
	@echo "Available targets:"
	@echo "  all           - Build all binaries"
	@echo "  ask-gpt2      - Build main GPT-2 binary"
	@echo "  ask-llama     - Build Llama Q&A binary"
	@echo "  generic-runner - Build generic architecture runner"
	@echo "  simple-demo   - Build simple tokenizer demo"
	@echo "  demo          - Run capital cities demo"
	@echo "  test          - Run tests"
	@echo "  fmt           - Format code"
	@echo "  vet           - Run go vet"
	@echo "  tidy          - Tidy dependencies"
	@echo "  clean         - Remove binaries"
	@echo "  help          - Show this help"
