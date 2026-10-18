"""M0 smoke test for one Local_Model: writes results/m0_smoke_<key>.json (task 1.4).

Usage: python scripts/smoke_test.py --model <key>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aib.smoke import main_smoke  # noqa: E402

if __name__ == "__main__":
    sys.exit(main_smoke())
