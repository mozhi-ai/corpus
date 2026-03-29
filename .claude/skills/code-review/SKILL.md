---
name: code-review
description: Review Python code for quality, ML best practices, and correctness
---

# Code Review Checklist

Review the code: $ARGUMENTS

## Check For

### Correctness
- Logic errors, off-by-one, wrong variable
- Edge cases: empty inputs, None values, zero-length sequences
- Numerical stability: overflow, underflow, division by zero

### Type Safety
- Type hints present and accurate
- No implicit Any types in function signatures
- Pydantic/dataclass validation at boundaries

### Performance
- Unnecessary copies of large tensors/arrays
- Missing batching for inference
- N+1 queries or repeated file I/O in loops
- Inefficient string concatenation in hot paths

### ML-Specific
- Data leakage between train/test
- Correct loss function for the task
- Proper gradient handling (detach, no_grad where needed)
- Reproducibility (seeds, deterministic ops)

### Security
- No hardcoded secrets or API keys
- No unsafe pickle/eval usage
- Input validation at API boundaries

## Output

Provide findings as a prioritized list: critical > major > minor > nit.
