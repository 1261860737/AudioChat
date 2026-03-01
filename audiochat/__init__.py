from __future__ import annotations

import sys
from pathlib import Path

_vendor_dir = Path(__file__).resolve().parents[1] / "third_party"
if _vendor_dir.exists():
    sys.path.insert(0, str(_vendor_dir))

_cosyvoice_dir = _vendor_dir / "CosyVoice"
if _cosyvoice_dir.exists():
    sys.path.insert(0, str(_cosyvoice_dir))
