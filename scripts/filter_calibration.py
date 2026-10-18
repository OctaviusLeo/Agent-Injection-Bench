"""Classifier check on ground-truth Tool_Outputs: writes results/m3_filter_calibration.json (task 4.3).

Usage: python scripts/filter_calibration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aib.calibration import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
