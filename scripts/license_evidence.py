"""Record the prior-work license evidence and the package license inventory (task 1.1).

Usage: python scripts/license_evidence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aib.evidence import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(Path(__file__).resolve().parent.parent))
