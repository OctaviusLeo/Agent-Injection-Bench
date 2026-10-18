"""Write the M1 catalog evidence to results/m1_catalog.json (task 2.3).

Usage: python scripts/catalog.py [--out <path>]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aib.catalog import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
