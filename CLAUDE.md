# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS336 Spring 2025 Assignment 1: Implementing fundamental language model components from scratch in PyTorch. This includes BPE tokenization, transformer architecture components, and training utilities.

## Commands

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_model.py

# Run a specific test
uv run pytest tests/test_model.py::test_linear

# Run Python files
uv run <python_file_path>

# Type checking
uv run ty check
```

## Architecture

### Implementation Pattern

This is an assignment codebase using the **adapter pattern**:
1. Students implement components in `cs336_basics/`
2. Wire implementations to tests via `tests/adapters.py`
3. Tests validate against reference implementations using snapshot testing

### Key Files

- **`tests/adapters.py`**: Bridge between tests and implementations. Contains 25+ function stubs that must be completed to connect custom implementations to the test suite.
- **`cs336_basics/bpe.py`**: BPE tokenizer training and pretokenization.
- **`tests/fixtures/`**: Reference data including GPT-2 vocab/merges, training corpora, and model weights.
- **`tests/_snapshots/`**: Reference outputs (`.npz` files) for snapshot testing with numerical tolerances.

### Components to Implement

**Tokenization Pipeline:**
- `train_bpe()` - BPE training with frequency-based merging
- `pretokenize()` - Text to pretokens using regex patterns
- Tokenizer encode/decode with special token handling

**Neural Network Layers:**
- Linear, Embedding, SiLU activation, RMSNorm
- SwiGLU feed-forward network (3-weight variant)

**Attention:**
- Scaled dot-product attention with masking
- Multi-head self-attention (batched, single matmul for all heads)
- RoPE (Rotary Position Embeddings)

**Full Model:**
- Pre-norm transformer blocks
- Complete transformer language model

**Training Infrastructure:**
- Batch sampling, cross-entropy loss, softmax
- AdamW optimizer with weight decay
- Cosine learning rate schedule with warmup
- Gradient clipping, checkpoint save/load

### Tensor Shape Conventions

Uses `jaxtyping` for explicit tensor dimensions:
```python
Float[Tensor, "batch sequence_length d_model"]
Int[Tensor, "... sequence_length"]
```

### Testing

Tests use snapshot comparisons against reference implementations stored in `tests/_snapshots/`. Numerical tolerance is applied (rtol, atol) for floating-point comparisons.
