from __future__ import annotations

import sys

from models.tracking import tracker as _tracker

sys.modules[__name__] = _tracker
