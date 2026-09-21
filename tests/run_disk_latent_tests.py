"""CPU-only DiskLatent and affected-node regressions; does not start a server."""
import os
from pathlib import Path
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.argv = ["disk-latent-tests", "--cpu", "--disable-xformers"]
import comfy.options
comfy.options.enable_args_parsing()
import comfy.model_management
sys.argv = ["disk-latent-tests"]

patterns = (
    "test_disk_audio*.py", "test_disk_latent*.py", "test_generated_wrappers*.py", "test_ksampler.py",
    "test_qwen_reference_latent_cache.py", "test_h3_loop_context.py",
    "test_h3_motion_context.py", "test_minimax_h3_masked_video_conditioning.py",
    "test_tiled_decode_colour_match.py",
)
suite = unittest.TestSuite(
    unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern=pattern)
    for pattern in patterns
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
