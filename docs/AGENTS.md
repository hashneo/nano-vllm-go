# Agent Development Guidelines

## Sandbox Environment Constraints

When developing or running agents/tools for this project, be aware of the following sandbox restrictions:

### Working Directory Requirements

**CRITICAL**: All operations must be performed within the project root directory.

- **Root Path**: `/Users/steven.taylor/Development/github/nano-vllm-go/`
- **Allowed Paths**: Only paths under the project root are accessible
- **Forbidden Paths**: `/tmp/`, `/private/tmp/`, or any system temp directories outside the project

### Temporary Files

If you need to create temporary files:

```bash
# ✓ CORRECT - Use project-local temp directory
mkdir -p ./tmp
echo "data" > ./tmp/temp_file.txt

# ✗ WRONG - System /tmp is not accessible
echo "data" > /tmp/temp_file.txt  # Will fail with "operation not permitted"
```

### Python Scripts

Python scripts that create temporary files must use project-relative paths:

```python
# ✓ CORRECT
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
temp_file = os.path.join(project_root, "tmp", "output.txt")

# ✗ WRONG
import tempfile
temp_file = tempfile.mktemp()  # May create file outside sandbox
```

### File Operations

All file operations (read, write, execute) must target paths within the project:

- Model files: `./models/`
- Build outputs: `./bin/`
- Temporary data: `./tmp/` (create if needed)
- Scripts: `./scripts/`
- Test outputs: `./output/` or `./tmp/`

### Environment Variables

The `TMPDIR` environment variable is set to `/tmp/claude/` in sandbox mode, but this is still restricted. Always use project-relative paths instead:

```bash
# ✓ CORRECT
OUTPUT_DIR="./tmp"
mkdir -p "$OUTPUT_DIR"

# ✗ WRONG
# Relying on $TMPDIR may fail
```

## Best Practices

1. **Always use relative paths** from project root (e.g., `./tmp/file.txt`)
2. **Create project-local temp directory** if needed: `mkdir -p ./tmp`
3. **Add to .gitignore**: Ensure `tmp/` is in `.gitignore` to avoid committing temporary files
4. **Clean up**: Remove temporary files after use with `rm -rf ./tmp/*`

## Example Workflows

### Running Tests with Output Files

```bash
# Create local output directory
mkdir -p ./tmp/test-results

# Run test and save output
go run ./cmd/test-something > ./tmp/test-results/output.txt

# Clean up when done
rm -rf ./tmp/test-results
```

### Python Model Conversions

```python
#!/usr/bin/env python3
import sys
import os

# Get project root (assuming script is in scripts/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(project_root, "tmp")
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "converted_model.bin")
# ... perform conversion ...
```

## Troubleshooting

### "Operation not permitted" errors

If you see errors like:
```
operation not permitted
Operation not permitted
```

Check that:
1. You're writing to paths within the project root
2. You're not using system `/tmp` or `/private/tmp`
3. File paths are relative to project root (e.g., `./tmp/file.txt` not `/tmp/file.txt`)

### Sandbox Violations

The sandbox logs violations. If operations fail unexpectedly:
1. Check the file path is within project root
2. Use `pwd` to verify current working directory is project root
3. Convert absolute paths to relative paths from project root
