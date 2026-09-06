import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import soundfile as sf
import torch


SOURCE = Path(__file__).resolve().parents[1] / "py" / "VTS_AudioChunks.py"
folder_paths = types.ModuleType("folder_paths")
management = types.ModuleType("comfy.model_management")
management.throw_exception_if_processing_interrupted = lambda: None
comfy = types.ModuleType("comfy")
comfy.model_management = management
with mock.patch.dict("sys.modules", {"folder_paths": folder_paths, "comfy": comfy, "comfy.model_management": management}):
    spec = importlib.util.spec_from_file_location("vts_audio_chunks_test_target", SOURCE)
    nodes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nodes)


def audio(values, rate=48000, channels=2):
    waveform = torch.as_tensor(values, dtype=torch.float32).reshape(1, 1, -1).repeat(1, channels, 1)
    return {"waveform": waveform, "sample_rate": rate}


class AudioChunksTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vts-audio-chunks-test-")
        self.root = Path(self.temporary.name)
        folder_paths.get_output_directory = lambda: str(self.root)
        self.block_patch = mock.patch.object(nodes, "BLOCK_FRAMES", 4)
        self.block_patch.start()

    def tearDown(self):
        self.block_patch.stop()
        self.temporary.cleanup()

    def save(self, values, chunks=None, **kwargs):
        return nodes.save_audio_chunk(audio(values, **kwargs), "audio/chunk", chunks)

    def test_exact_float32_round_trip_order_and_no_input_manifest_mutation(self):
        first = audio([-0.0, 0.0, 1e-40, -1e-30, 1.25, -2.0, 0.7])
        second = audio([99.0, -0.1, 3.0])
        one, _ = nodes.save_audio_chunk(first)
        saved_one = dict(one)
        two, _ = nodes.save_audio_chunk(second, chunks=one)
        path, assembled = nodes.assemble_audio_chunks(two, return_audio=True)
        expected = torch.cat([first["waveform"], second["waveform"]], dim=-1)
        self.assertTrue(torch.equal(expected.view(torch.int32), assembled["waveform"].view(torch.int32)))
        self.assertEqual(assembled["sample_rate"], 48000)
        self.assertEqual(one, saved_one)
        self.assertEqual(two["frames"], expected.shape[-1])
        self.assertEqual(two["chunks"], 2)
        info = sf.info(path)
        self.assertEqual((info.format, info.subtype, info.frames), ("RF64", "FLOAT", 10))

    def test_file_only_streams_without_materializing_an_audio_tensor(self):
        one, _ = self.save(range(17))
        two, _ = self.save(range(19), one)
        observed_blocks = []
        original = sf.SoundFile.write

        def checked_write(sink, block):
            observed_blocks.append(len(block))
            return original(sink, block)

        with mock.patch.object(sf, "read", side_effect=AssertionError("Full-file read forbidden")), \
                mock.patch.object(torch, "from_numpy", side_effect=AssertionError("Tensor allocation forbidden")), \
                mock.patch.object(sf.SoundFile, "write", checked_write):
            path, output = nodes.assemble_audio_chunks(two)
        self.assertIsNone(output)
        self.assertEqual(sf.info(path).frames, 36)
        self.assertLessEqual(max(observed_blocks), 4)

    def test_descriptor_stays_small_and_contains_no_tensor_history(self):
        chunks = None
        for index in range(15):
            chunks, _ = self.save([index], chunks)
        self.assertLess(len(json.dumps(chunks)), 350)
        self.assertEqual(set(chunks), {"version", "manifest", "sample_rate", "channels", "frames", "chunks"})
        path, _ = nodes.assemble_audio_chunks(chunks)
        data, _ = sf.read(path, dtype="float32", always_2d=True)
        self.assertEqual(data[:, 0].tolist(), list(range(15)))

    def test_sample_rate_and_channel_mismatches_are_rejected_before_writing(self):
        one, path = self.save([1, 2])
        existing = set(self.root.rglob("*"))
        for kwargs in ({"rate": 44100}, {"channels": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "sample rate and channel"):
                self.save([3, 4], one, **kwargs)
        self.assertEqual(set(self.root.rglob("*")), existing)
        self.assertTrue(Path(path).exists())

    def test_path_traversal_absolute_windows_and_unc_paths_rejected(self):
        for prefix in ("../escape", "audio/../../escape", "/tmp/escape", "C:\\escape", "\\\\host\\share\\escape", "audio//name", "audio/./name"):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                nodes.save_audio_chunk(audio([1]), prefix)

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="vts-audio-outside-") as outside:
            (self.root / "escape").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "outside"):
                nodes.save_audio_chunk(audio([1]), "escape/chunk")
            self.assertEqual(list(Path(outside).iterdir()), [])

    def test_resolved_output_root_can_itself_be_a_mapped_or_symlinked_directory(self):
        mapped = self.root / "mounted_share"
        mapped.mkdir()
        alias = self.root / "output_alias"
        alias.symlink_to(mapped, target_is_directory=True)
        folder_paths.get_output_directory = lambda: str(alias)
        chunks, path = self.save([1, 2, 3])
        self.assertTrue(Path(path).is_relative_to(mapped))
        assembled, _ = nodes.assemble_audio_chunks(chunks)
        self.assertTrue(Path(assembled).is_relative_to(mapped))

    def test_same_prefix_never_overwrites_existing_outputs(self):
        one, first = self.save([1, 2])
        original = Path(first).read_bytes()
        _, second = self.save([8, 9])
        self.assertNotEqual(first, second)
        self.assertEqual(Path(first).read_bytes(), original)
        a, _ = nodes.assemble_audio_chunks(one)
        b, _ = nodes.assemble_audio_chunks(one)
        self.assertNotEqual(a, b)
        self.assertTrue(Path(a).exists() and Path(b).exists())

    def test_cancellation_removes_only_the_current_partial_save(self):
        _, first = self.save([1, 2])
        original = Path(first).read_bytes()
        directories = set((self.root / "audio").iterdir())
        calls = 0

        def interrupt():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise InterruptedError("cancelled")

        with mock.patch.object(management, "throw_exception_if_processing_interrupted", interrupt):
            with self.assertRaises(InterruptedError):
                self.save(range(20))
        self.assertEqual(set((self.root / "audio").iterdir()), directories)
        self.assertEqual(Path(first).read_bytes(), original)

    def test_assembly_write_failure_cleans_partial_but_preserves_chunks(self):
        chunks, path = self.save(range(12))
        original = Path(path).read_bytes()
        directories = set((self.root / "audio").iterdir())
        with mock.patch.object(sf.SoundFile, "write", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                nodes.assemble_audio_chunks(chunks)
        self.assertEqual(set((self.root / "audio").iterdir()), directories)
        self.assertEqual(Path(path).read_bytes(), original)

    def test_tampered_manifest_and_chunk_file_are_rejected(self):
        chunks, path = self.save([1, 2, 3])
        forged = dict(chunks, frames=900)
        with self.assertRaisesRegex(ValueError, "descriptor disagrees"):
            nodes.assemble_audio_chunks(forged)
        with self.assertRaises(ValueError):
            nodes.assemble_audio_chunks(dict(chunks, manifest="../manifest.json"))
        sf.write(path, [[1.0], [2.0]], 44100, format="RF64", subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "file disagrees"):
            nodes.assemble_audio_chunks(chunks)

    def test_manifest_chain_order_is_checked(self):
        first, _ = self.save([1])
        second, _ = self.save([2], first)
        path = self.root / second["manifest"]
        record = json.loads(path.read_text())
        record["previous"] = second["manifest"]
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "cycle|inconsistent"):
            nodes.assemble_audio_chunks(second)

    def test_non_object_manifest_has_a_clear_validation_error(self):
        chunks, _ = self.save([1, 2, 3])
        (self.root / chunks["manifest"]).write_text("[]")
        with self.assertRaisesRegex(ValueError, "manifest version"):
            nodes.assemble_audio_chunks(chunks)

    def test_publication_does_not_replace_an_existing_file(self):
        final = self.root / "existing.wav"
        temporary = self.root / "partial.wav"
        final.write_bytes(b"existing user output")
        temporary.write_bytes(b"new data")
        with self.assertRaises(FileExistsError):
            nodes._publish(temporary, final)
        self.assertEqual(final.read_bytes(), b"existing user output")
        self.assertTrue(temporary.exists())

    def test_unsupported_audio_batch_dtype_and_rate(self):
        for value in ({"waveform": torch.zeros(2, 2, 3), "sample_rate": 48000},
                      {"waveform": torch.zeros(1, 2, 3, dtype=torch.float64), "sample_rate": 48000},
                      {"waveform": torch.zeros(1, 2, 3), "sample_rate": 0}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                nodes.save_audio_chunk(value)


if __name__ == "__main__":
    unittest.main()
