"""CI smoke subset with the Stub_Model (task 6.3): writes runs/ci_smoke.json, exits non-zero on a mismatch.

Usage: uv run --frozen python scripts/ci_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aib.ci_smoke import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
