"""vendored TTT/MAG components copied from sparse-linear-attention."""

# keep this package import-light because the vendored submodules pull in
# optional `fla` dependencies that are not always installed in the EM-LLM env.
__all__ = ["LinearSGD", "MemoryAsGate", "TTT"]
