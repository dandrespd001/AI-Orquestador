from __future__ import annotations

import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"id": request.get("id"), "result": {"ok": True}}), flush=True)
