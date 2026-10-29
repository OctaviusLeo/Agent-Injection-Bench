"""Claude check: a tool-free request, a tool request, and the tool result back to the Claude target (claude-provider Req C4.2).

Usage: python scripts/claude_check.py   (needs the key named in config/models.yaml; writes results/claude_check.json)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aib.claude import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
