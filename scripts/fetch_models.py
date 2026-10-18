"""Download the pinned model files into ~/aib-models and write results/m0_models.json (task 1.3).

Usage: python scripts/fetch_models.py [--model <key>|classifier]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aib.fetch import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
