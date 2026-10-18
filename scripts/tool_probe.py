"""M0 tool-call probe for one Local_Model (server running): writes results/m0_tool_probe_<key>.json (task 1.4).

Usage: python scripts/tool_probe.py --model <key> [--attempt N]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aib.smoke import main_probe  # noqa: E402

if __name__ == "__main__":
    sys.exit(main_probe())
