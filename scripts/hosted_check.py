"""M0 hosted check: one tool-free and one tool request to the hosted model (task 1.5).

Usage: python scripts/hosted_check.py   (needs OPENAI_API_KEY; writes results/m0_hosted_check.json)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aib.hosted import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
