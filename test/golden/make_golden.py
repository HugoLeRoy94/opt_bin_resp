"""Re-record test/golden/*.json from THIS code.

    python3 test/golden/make_golden.py

Only after a deliberate change of the physics or of an estimator. Any other
change that moves a number is a bug, and re-recording would hide it.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from configs import CELL, CELL_MEASURE, RECEPTOR, RECEPTOR_MEASURE   # noqa: E402
from test_golden import _run                                          # noqa: E402

for name, fields, measure in [("receptor", RECEPTOR, RECEPTOR_MEASURE),
                              ("cell", CELL, CELL_MEASURE)]:
    (HERE / f"{name}.json").write_text(json.dumps(_run(fields, measure), indent=4))
    print(f"wrote {name}.json")
