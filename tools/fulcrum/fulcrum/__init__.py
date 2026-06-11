"""fulcrum — a causal performance-decision engine.

Not a profiler (profilers attribute; attribution manufactures phantom levers) and
not a benchmark harness (numbers without provenance produce phantom comparisons).
Fulcrum closes the attribution -> causal gap and outputs ranked, re-verifiable
actions, each carrying either a tool-executed causal A/B verdict or an explicit
HYPOTHESIS label with the exact perturbation that would test it.

See plans/fulcrum-product.md (charter) in the host repo, and README.md here.
"""

__version__ = "3.0.0"

# The measurement-protocol version. Bumped whenever the meaning of a stored
# number changes (sink law, mask convention, timing method). Part of every
# measurement fingerprint: numbers from different protocols never compare.
PROTOCOL_VERSION = "fulcrum-v3"
