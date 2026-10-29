import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "tiny_llm_from_scratch"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.generation.artifacts import ClaudeArtifactEngine

sample = """Here is the Python implementation:

```python
def calculate_prime_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors
```

And here is the architecture diagram:

```mermaid
graph TD
    Client --> CLI
    CLI --> Orchestrator
    Orchestrator --> DuckDuckGo
    Orchestrator --> Artifacts
```
"""

artifacts = ClaudeArtifactEngine.extract_artifacts(sample)
print(f"Extracted Artifacts: {len(artifacts)}")
for a in artifacts:
    print(ClaudeArtifactEngine.format_artifact_display(a))
    print()
