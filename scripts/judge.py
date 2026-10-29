"""Second-opinion judge: Claude checks each classified attack case of a Results_Store (claude-provider Req C3).

Usage: python scripts/judge.py --store results/raw/<model>.jsonl [--limit N]
(needs the key named in config/models.yaml; writes results/judge/<model>.jsonl and results/judge/<model>_agreement.json)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aib.judge import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
