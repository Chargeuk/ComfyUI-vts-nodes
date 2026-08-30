from __future__ import annotations

import importlib.util
from pathlib import Path


node_path = Path(__file__).resolve().parents[1] / "py" / "VTS_Rectilinear_To_VR180.py"
spec = importlib.util.spec_from_file_location("vts_vr180_import_test", node_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert sorted(module.NODE_CLASS_MAPPINGS) == [
    "VTSRectilinearToVR180",
    "VTSVR180ToRectilinear",
]
print("node import: PASS")
