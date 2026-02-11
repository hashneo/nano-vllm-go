# SSM Debugging Guide

## Quick Start

To verify the SSM is working correctly:

```bash
# Run the demonstration
go run ./cmd/demo-ssm-correct/main.go
```

## Debugging Tools

### 1. Token-by-Token SSM Testing
**Purpose**: Verify SSM computation matches PyTorch for each token in a sequence

```bash
go run ./cmd/debug-ssm-sequence/main.go
```

**What it does**:
- Loads model and PyTorch reference data
- Processes 5 tokens sequentially with state persistence
- Compares SSM output for each token with PyTorch
- Shows state statistics after each token
- Identifies if/where divergence occurs

**Output**: Detailed comparison for each token showing:
- Input values (x, B, C, delta)
- SSM outputs (Go vs PyTorch)
- Statistics (max diff, mean diff, sign mismatches)
- State accumulation metrics

### 2. Full Pipeline Step-by-Step Testing
**Purpose**: Verify each post-SSM processing step

```bash
go run ./cmd/debug-mamba-full/main.go
```

**What it does**:
- Tests gate activation (SiLU)
- Tests gating operation (SSM * gate)
- Tests RMS normalization
- Tests output projection
- Compares each step with PyTorch reference values

**Output**: For each step shows:
- Go vs PyTorch values
- Comparison statistics
- Pass/fail status

### 3. Indexing Verification
**Purpose**: Verify multi-token sequence indexing is correct

```bash
go run ./cmd/test-gating-index/main.go
```

**What it does**:
- Tests the gating indexing formula
- Verifies correct mapping for all tokens
- Shows which indices map to which tokens/dimensions

**Output**: Index mappings and verification results

### 4. Correctness Demonstration
**Purpose**: Show comprehensive test results

```bash
go run ./cmd/demo-ssm-correct/main.go
```

**What it does**:
- Displays model configuration
- Shows SSM parameters
- Lists all verification results
- Explains mathematical correctness
- Provides conclusion

## Generating Reference Data

### Generate SSM Inputs for All Tokens
```bash
python3 scripts/save_ssm_inputs_all_tokens.py
```

**Output**: `ssm_inputs_all_tokens.json`
- Contains x, B, C, delta for each token
- Used by `debug-ssm-sequence` to test SSM computation

### Generate SSM Outputs for All Tokens
```bash
python3 scripts/save_ssm_outputs_all_tokens.py
```

**Output**: `ssm_outputs_all_tokens.json`
- Contains SSM outputs for each token from PyTorch
- Used for comparison with Go implementation

### Generate All Intermediate Values
```bash
python3 scripts/save_mamba_intermediates.py
```

**Output**: `mamba_intermediates.json`
- Contains gate values
- Contains gate-activated values
- Contains SSM outputs
- Contains post-gating values
- Contains post-normalization values
- Contains final outputs
- Used by `debug-mamba-full` for step-by-step testing

## Understanding the Output

### Sign Mismatches
- **0.0%**: Perfect match (values have same sign)
- **< 1%**: Excellent match (minor numerical differences)
- **1-5%**: Good match (acceptable for floating point)
- **> 5%**: Divergence (investigate)

### Max/Mean Absolute Difference
- **< 0.00001**: Perfect match
- **< 0.001**: Excellent match
- **< 0.01**: Good match
- **> 0.01**: Moderate divergence
- **> 0.1**: Significant divergence

### State Statistics
- **Non-zero elements**: Should be > 95% for active processing
- **Mean abs value**: Typically 0.001 - 0.01 range
- **Max abs value**: Should be reasonable (< 1.0 for stable states)

## Test Sequence

All tests use the same 5-token sequence:
```
Tokens: [791, 6864, 315, 10057, 374]
Text: "The capital of Germany is"
```

## File Structure

```
nano-vllm-go/
├── cmd/
│   ├── debug-ssm-sequence/main.go       # Token-by-token testing
│   ├── debug-mamba-full/main.go         # Full pipeline testing
│   ├── test-gating-index/main.go        # Index verification
│   └── demo-ssm-correct/main.go         # Demonstration
├── scripts/
│   ├── save_ssm_inputs_all_tokens.py    # Generate SSM inputs
│   ├── save_ssm_outputs_all_tokens.py   # Generate SSM outputs
│   └── save_mamba_intermediates.py      # Generate intermediates
├── purego/tensor/
│   └── mamba2.go                        # SSM implementation
├── ssm_inputs_all_tokens.json           # Reference inputs
├── ssm_outputs_all_tokens.json          # Reference outputs
├── mamba_intermediates.json             # Intermediate values
├── SSM_DEBUG_SUMMARY.md                 # Summary of findings
├── SSM_DEBUG_COMPLETE_REPORT.md         # Detailed report
└── SSM_DEBUGGING_GUIDE.md               # This file
```

## Common Issues and Solutions

### Issue: "Failed to load reference data"
**Solution**: Run the Python scripts to generate reference data first

### Issue: "Model not found"
**Solution**: Ensure `./models/granite-350m` exists and contains the model files

### Issue: Different results from documentation
**Solution**: Regenerate reference data with current PyTorch model:
```bash
python3 scripts/save_ssm_inputs_all_tokens.py
python3 scripts/save_ssm_outputs_all_tokens.py
python3 scripts/save_mamba_intermediates.py
```

## Debugging Workflow

If you observe divergence in production:

1. **Generate fresh reference data**:
   ```bash
   python3 scripts/save_ssm_inputs_all_tokens.py
   python3 scripts/save_ssm_outputs_all_tokens.py
   ```

2. **Test SSM computation**:
   ```bash
   go run ./cmd/debug-ssm-sequence/main.go
   ```
   - If SSM matches, problem is elsewhere
   - If SSM diverges, investigate state management

3. **Test full pipeline**:
   ```bash
   go run ./cmd/debug-mamba-full/main.go
   ```
   - Identifies which step causes divergence
   - Compare with expected intermediate values

4. **Check inputs**:
   - Compare embeddings with PyTorch
   - Compare in_proj outputs
   - Compare conv1d outputs

## Mathematical Background

### SSM State Update
```
x[n+1] = A_bar * x[n] + dt * B * u[n]
```

Where:
- `A_bar = exp(A * dt)` is the discrete-time state transition matrix
- `A = -exp(A_log)` is the continuous-time decay matrix
- `dt` is the time step (computed dynamically)
- `B` is the input matrix
- `u[n]` is the input at timestep n

### SSM Output
```
y[n] = C * x[n] + D * u[n]
```

Where:
- `C` is the output matrix
- `x[n]` is the state at timestep n
- `D` is the skip connection
- `u[n]` is the input at timestep n

## Support

For questions or issues:
1. Check the documentation in `SSM_DEBUG_SUMMARY.md`
2. Run the debugging tools to isolate the problem
3. Compare intermediate values with PyTorch
4. Check the implementation in `purego/tensor/mamba2.go`

## Verification Checklist

- [x] SSM computation matches PyTorch
- [x] State persistence works correctly
- [x] Gate activation is correct
- [x] Gating operation is correct
- [x] RMS normalization is correct
- [x] Output projection is correct
- [x] Multi-token indexing is correct
- [x] All 5 tokens match exactly

**Status**: ✅ ALL VERIFIED
