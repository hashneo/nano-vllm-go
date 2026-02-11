.PHONY: all build clean test fmt vet tidy help

# Build directory
BIN_DIR=bin

# Main binaries
all: ask generic-runner simple-demo

# Build unified ask CLI
ask:
	@echo "Building unified ask CLI..."
	@mkdir -p $(BIN_DIR)
	@go build -o $(BIN_DIR)/ask ./cmd/ask

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
demo: ask
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
	@echo "  all            - Build all binaries"
	@echo "  ask            - Build unified ask CLI (supports gpt2/llama/falcon/granite)"
	@echo "  generic-runner - Build generic architecture runner"
	@echo "  simple-demo    - Build simple tokenizer demo"
	@echo "  demo           - Run capital cities demo"
	@echo "  test           - Run tests"
	@echo "  fmt            - Format code"
	@echo "  vet            - Run go vet"
	@echo "  tidy           - Tidy dependencies"
	@echo "  clean          - Remove binaries"
	@echo "  help           - Show this help"
