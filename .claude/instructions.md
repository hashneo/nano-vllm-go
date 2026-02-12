# Project Instructions for Claude

## Required Reading

Before working on this project, please read the following documentation:

- **Agent Development Guidelines**: `docs/agents.md` - Contains critical sandbox environment constraints and best practices

## Sandbox Environment

This project operates in a **sandboxed environment** with strict filesystem restrictions:

- All file operations MUST use paths relative to the project root
- Use `./` prefix for all paths (e.g., `./tmp/`, `./models/`, `./bin/`)
- NEVER use absolute paths or system temp directories like `/tmp/`
- NEVER reference the user's home directory or absolute project path

## Path Requirements

**CORRECT Examples:**
```bash
mkdir -p ./tmp
echo "data" > ./tmp/temp_file.txt
go build -o ./bin/myapp ./cmd/myapp
```

**INCORRECT Examples:**
```bash
# DO NOT use absolute paths
echo "data" > /tmp/temp_file.txt
echo "data" > /Users/username/project/file.txt

# DO NOT reference user paths
cd ~/Development/project
```

## Key Directories (all relative to project root)

- `./models/` - Model files
- `./bin/` - Build outputs
- `./tmp/` - Temporary data (create if needed)
- `./scripts/` - Utility scripts
- `./docs/` - Documentation
- `./cmd/` - Command line applications
- `./purego/` - Pure Go implementation

## Important Notes

1. Always use relative paths from project root
2. Create `./tmp/` directory if temporary files are needed
3. Clean up temporary files after use
4. Refer to `docs/agents.md` for complete sandbox constraints and troubleshooting

This ensures all operations work correctly within the sandbox and maintain portability across different systems.
