"""CPU contracts; opt in to small CUDA kernel tests with VTS_HYBRID_CUDA_TEST=1."""

import gc
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
import weakref

import torch
import torch.nn.functional as F


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
VDN = Path(os.environ.get("VTS_VDN_ROOT", str(ROOT.parent / "ComfyUI-VDN-H3")))
NODE = load("vts_hybrid_test_node", ROOT / "py" / "VTS_H3HybridAttention.py")
WINDOW = load("vts_hybrid_test_window", VDN / "vdn_h3" / "window.py")


def exact(q, k, v, scale, transformer_options=None):
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None],
        v.transpose(0, 1)[None], scale=scale)
    return out[0].transpose(0, 1)


def make_patch(**kwargs):
    settings = dict(sparsity=0.70, min_tokens=0, dense_first_steps=1,
                    dense_last_steps=1, dense_blocks=set(), fallback_to_exact=True,
                    verbose=False, block_map=mock.Mock(), kernel=mock.Mock(),
                    exact_attention=exact)
    settings.update(kwargs)
    return NODE.HybridAttention(**settings)


class WindowHookTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        WINDOW.clear_window_state()
        self.sdpa = mock.patch.object(WINDOW, "_sdpa", exact)
        self.sdpa.start()

    def tearDown(self):
        self.sdpa.stop()
        WINDOW.clear_window_state()

    def run_window(self, anchor_mode, hook=None, prefix=3, suffix=2, chunk=0):
        frames, per_frame = 9, 3
        video_end = prefix + frames * per_frame
        q, k, v = [torch.randn(video_end + suffix, 2, 8) for _ in range(3)]
        args = (q, k, v, prefix, video_end, frames, per_frame,
                WINDOW.window_bounds(frames, 1, chunk), 8 ** -0.5)
        plain = WINDOW.window_softmax_grouped(*args, anchor_frames=anchor_mode,
                                             retain_buffers=False)
        patched = WINDOW.window_softmax_grouped(*args, anchor_frames=anchor_mode,
                                               retain_buffers=False, local_attention=hook)
        return plain, patched, args

    def test_exact_hook_preserves_all_anchor_modes_and_window_geometry(self):
        for mode in WINDOW.ANCHOR_FRAME_MODES:
            for chunk in (0, 2):
                with self.subTest(mode=mode, chunk=chunk):
                    plain, patched, _ = self.run_window(
                        mode, lambda q, k, v, scale, ranges: exact(q, k, v, scale), chunk=chunk)
                    torch.testing.assert_close(patched, plain, rtol=0, atol=0)

    def test_hook_never_changes_global_or_anchor_queries(self):
        calls = []
        def zero(q, k, v, scale, ranges):
            calls.append((q.shape[0], ranges))
            return torch.zeros_like(q)
        plain, patched, args = self.run_window("both", zero)
        start, end = args[3:5]
        for a, b in ((0, start), (end, len(plain)), (start, start + 3), (end - 3, end)):
            torch.testing.assert_close(patched[a:b], plain[a:b], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(patched[start + 3:end - 3]), 0)
        self.assertTrue(calls)

    def test_protected_ranges_use_gathered_key_coordinates(self):
        for mode in WINDOW.ANCHOR_FRAME_MODES:
            with self.subTest(mode=mode):
                frames, per_frame, start, end, seq = 9, 3, 3, 30, 32
                bounds = WINDOW.window_bounds(frames, 1, 2)
                plan = WINDOW._window_plan(start, end, frames, per_frame, bounds,
                                           mode, seq, torch.device("cpu"))
                for (_, win_idx), ranges in zip(plan["groups"], plan["group_protected_ranges"]):
                    gathered = torch.cat((plan["global_idx"], win_idx)).tolist()
                    protected = {gathered[i] for a, b in ranges for i in range(a, b)}
                    expected = set(range(start)) | set(range(end, seq))
                    if mode in ("columns", "both"):
                        expected |= set(range(start, start + per_frame))
                        expected |= set(range(end - per_frame, end))
                    self.assertEqual(protected, expected)

    def test_no_global_tokens(self):
        plain, patched, _ = self.run_window(
            "none", lambda q, k, v, scale, ranges: exact(q, k, v, scale), prefix=0, suffix=0)
        torch.testing.assert_close(patched, plain, rtol=0, atol=0)


class PolicyTests(unittest.TestCase):
    def test_disabled_and_zero_sparsity_are_identity_without_dependencies(self):
        model = object()
        node = NODE.VTS_H3HybridAttention()
        self.assertIs(node.execute(model, enabled=False)[0], model)
        self.assertIs(node.execute(model, local_sparsity=0)[0], model)

    def test_unpatched_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "Apply VDN-H3"):
            NODE.VTS_H3HybridAttention().execute(mock.Mock(object_patches={}))

    def test_node_clones_model_and_rejects_duplicate(self):
        class Model:
            def __init__(self):
                self.object_patches = {"attn.forward": types.SimpleNamespace(_vdn_forward=True)}
                self.wrappers = {}
            def clone(self):
                cloned = Model()
                cloned.wrappers = {k: v.copy() for k, v in self.wrappers.items()}
                return cloned
            def add_wrapper_with_key(self, kind, key, wrapper):
                self.wrappers.setdefault(kind, {})[key] = [wrapper]
            def get_wrappers(self, kind, key):
                return self.wrappers.get(kind, {}).get(key, [])
        vdn = types.ModuleType("vdn_h3")
        vdn.hybrid = types.SimpleNamespace(VDN_LOCAL_ATTENTION_API=1)
        vdn.window = types.SimpleNamespace(_sdpa=exact)
        extension = types.ModuleType("comfy.patcher_extension")
        extension.WrappersMP = types.SimpleNamespace(OUTER_SAMPLE="outer_sample", DIFFUSION_MODEL="diffusion_model")
        original = Model()
        with mock.patch.dict(sys.modules, {"vdn_h3": vdn, "comfy.patcher_extension": extension}), \
                mock.patch.object(NODE, "_load_sla_backend", return_value=(mock.Mock(), mock.Mock())):
            result = NODE.VTS_H3HybridAttention().execute(original)[0]
            self.assertIsNot(result, original)
            self.assertEqual(original.wrappers, {})
            self.assertEqual(set(result.wrappers), {"outer_sample", "diffusion_model"})
            with self.assertRaisesRegex(ValueError, "already applied"):
                NODE.VTS_H3HybridAttention().execute(result)
            vdn.hybrid.VDN_LOCAL_ATTENTION_API = 0
            with self.assertRaisesRegex(RuntimeError, "restart ComfyUI"):
                NODE.VTS_H3HybridAttention().execute(original)

    def test_cpu_stays_exact_without_loading_kernel(self):
        patch = make_patch()
        q, k, v = torch.randn(3, 2, 128), torch.randn(7, 2, 128), torch.randn(7, 2, 128)
        torch.testing.assert_close(patch.attend(q, k, v, 0.1, []), exact(q, k, v, 0.1), rtol=0, atol=0)
        patch.kernel.assert_not_called()

    def test_schedule_resolves_logical_steps_not_call_count(self):
        schedule = [1, .8, .6, .4, .2, 0]
        self.assertEqual(NODE._sampling_step(dict(sample_sigmas=schedule, sigmas=[.4])), (3, 5))
        self.assertEqual(NODE._sampling_step(dict(sample_sigmas=torch.tensor(schedule), sigmas=torch.tensor([.4]))), (3, 5))
        self.assertIsNone(NODE._sampling_step({}))

    def test_first_last_and_missing_metadata_are_exact(self):
        patch = make_patch()
        schedule = [1, .75, .5, .25, 0]
        for sigma, active in ((1, False), (.75, True), (.75, True), (.5, True), (.25, False)):
            options = dict(sample_sigmas=schedule, sigmas=[sigma], unrelated=123)
            def executor(*args, **kw):
                result = args[3]
                self.assertEqual("vdn_local_attention" in result, active)
                self.assertEqual(result["unrelated"], 123)
                return "ok"
            self.assertEqual(patch.wrap_diffusion(executor, None, None, None, options), "ok")
            self.assertNotIn("vdn_local_attention", options)
        options = {}
        patch.wrap_diffusion(lambda **kw: self.assertNotIn("vdn_local_attention", kw["transformer_options"]),
                             transformer_options=options)

    def test_no_step_guards_allow_custom_sampler(self):
        patch = make_patch(dense_first_steps=0, dense_last_steps=0)
        patch.wrap_diffusion(lambda **kw: self.assertIn("vdn_local_attention", kw["transformer_options"]))

    def test_layer_exclusions(self):
        patch = make_patch(dense_blocks=NODE._dense_blocks("0-2, 49"))
        self.assertIsNone(patch.for_block(0))
        self.assertIsNone(patch.for_block(49))
        self.assertIsNotNone(patch.for_block(3))

    def test_cleanup_after_success_and_interrupt(self):
        patch = make_patch()
        def executor(fail=False):
            patch.failed = True
            patch.sparse_calls = 12
            if fail:
                raise RuntimeError("interrupted")
            return 7
        self.assertEqual(patch.wrap_sample(executor), 7)
        self.assertFalse(patch.failed)
        self.assertEqual(patch.sparse_calls, 0)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            patch.wrap_sample(executor, fail=True)
        self.assertFalse(patch.failed)
        self.assertEqual(patch.sparse_calls, 0)

    def test_no_tensor_retention(self):
        patch = make_patch()
        q, k, v = [torch.randn(5, 2, 128) for _ in range(3)]
        refs = [weakref.ref(t) for t in (q, k, v)]
        result = patch.attend(q, k, v, .1, [])
        del q, k, v, result
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertFalse(any(isinstance(item, torch.Tensor) for item in vars(patch).values()))


@unittest.skipUnless(os.environ.get("VTS_HYBRID_CUDA_TEST") == "1" and torch.cuda.is_available(),
                     "set VTS_HYBRID_CUDA_TEST=1 for isolated GPU checks")
class CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block_map, cls.kernel = NODE._load_sla_backend()

    def make_gpu_patch(self, **kwargs):
        settings = dict(block_map=type(self).block_map, kernel=type(self).kernel)
        settings.update(kwargs)
        return make_patch(**settings)

    def inputs(self):
        torch.manual_seed(88)
        return [torch.randn(n, 2, 128, device="cuda", dtype=torch.bfloat16)
                for n in (131, 777, 777)]

    def test_rectangular_kernel_matches_selected_key_sdpa(self):
        q, k, v = self.inputs()
        ranges = [(0, 67), (650, 777)]
        qb, kb, vb = (t[None] for t in (q, k, v))
        lut, topk = type(self).block_map(qb, kb, .2, 64, 64, protect_ranges=ranges)
        out = type(self).kernel(qb, kb, vb, lut, topk, 64, 64)[0]
        self.assertLess(topk, 13)
        mask = torch.zeros(1, 2, len(q), len(k), dtype=torch.bool, device="cuda")
        tables = lut.cpu().tolist()[0]
        for head, rows in enumerate(tables):
            for row, columns in enumerate(rows):
                self.assertTrue({0, 1, 10, 11, 12}.issubset(columns))
                for column in columns:
                    mask[0, head, row * 64:(row + 1) * 64, column * 64:(column + 1) * 64] = True
        expected = F.scaled_dot_product_attention(q.transpose(0, 1)[None].float(),
                   k.transpose(0, 1)[None].float(), v.transpose(0, 1)[None].float(), attn_mask=mask)[0].transpose(0, 1)
        torch.testing.assert_close(out.float(), expected, rtol=.03, atol=.004)
        self.assertTrue(torch.isfinite(out).all())

    def test_real_hybrid_path_and_protection_saturation(self):
        q, k, v = self.inputs()
        patch = self.make_gpu_patch(sparsity=.8)
        result = patch.attend(q, k, v, 128 ** -.5, [(0, 64)])
        self.assertEqual(patch.sparse_calls, 1)
        self.assertEqual(result.shape, q.shape)
        self.assertTrue(torch.isfinite(result).all())
        result = patch.attend(q, k, v, 128 ** -.5, [(0, len(k))])
        torch.testing.assert_close(result, exact(q, k, v, 128 ** -.5), rtol=0, atol=0)
        self.assertEqual(patch.exact_calls, 1)

    def test_recoverable_failure_falls_back_but_oom_propagates(self):
        q, k, v = self.inputs()
        patch = self.make_gpu_patch(kernel=mock.Mock(side_effect=RuntimeError("unsupported kernel")))
        result = patch.attend(q, k, v, .1, [])
        torch.testing.assert_close(result, exact(q, k, v, .1), rtol=0, atol=0)
        self.assertTrue(patch.failed)
        patch = self.make_gpu_patch(kernel=mock.Mock(side_effect=torch.OutOfMemoryError("test OOM")))
        with self.assertRaises(torch.OutOfMemoryError):
            patch.attend(q, k, v, .1, [])


if __name__ == "__main__":
    unittest.main()
