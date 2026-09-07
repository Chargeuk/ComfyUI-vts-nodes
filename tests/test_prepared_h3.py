"""Small CPU round trips through real ComfyUI H3 loading and checkpoint writing.

Run from ComfyUI with its Python; no full H3 checkpoint or GPU required.
"""

from copy import deepcopy
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("VTS_COMFY_ROOT", Path.cwd()))
sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(COMFY / "custom_nodes/ComfyUI-VDN-H3"))
from comfy.cli_args import args
args.cpu = True
import comfy.model_management
import comfy.sd
import comfy.supported_models
import folder_paths
from vdn_h3 import spec as vdn_spec

spec = importlib.util.spec_from_file_location("prepared_h3_test", ROOT / "py/VTS_PreparedH3.py")
NODE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NODE)


@pytest.fixture(autouse=True)
def folders(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", deepcopy(folder_paths.folder_names_and_paths))
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path / "output"))
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path / "models"))
    folder_paths.folder_names_and_paths["diffusion_models"] = (
        [str(tmp_path / "models/unet"), str(tmp_path / "models/diffusion_models")], {".safetensors"})
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    yield
    comfy.model_management.unload_all_models()
    gc.collect()


def tiny_model(quantized=False):
    config = comfy.supported_models.MiniMaxH3(dict(image_model="minimax_h3", hidden_size=32,
        num_layers=2, token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=16,
        ffn_hidden_size=64, latents_dim=4, audio_latents_dim=4, text_dim=32,
        timestep_input_dim=16, time_embed_hidden_size=32, time_embed_dim=16, rope_inv_freq_len=2))
    config.set_inference_dtype(torch.float32, None)
    raw = config.get_model({}, device=torch.device("cpu"))
    torch.manual_seed(70)
    sd = {key: torch.randn_like(value) * 0.05 for key, value in raw.diffusion_model.state_dict().items()}
    if quantized:
        prefix = "blocks.0.attn.qkv_proj."
        dtype = torch.float8_e4m3fn if quantized == "float8_e4m3fn" else torch.int8
        sd[prefix + "weight"] = torch.randint(-40, 40, sd[prefix + "weight"].shape).to(dtype)
        sd[prefix + "weight_scale"] = torch.tensor(0.005)
        conf = json.dumps({"format": quantized if isinstance(quantized, str) else "int8_tensorwise"})
        sd[prefix + "comfy_quant"] = torch.tensor(list(conf.encode()), dtype=torch.uint8)
    return comfy.sd.load_diffusion_model_state_dict(sd)


def with_vdn(tmp_path, model):
    shapes = {"to_out_linear.weight": (32, 32), "beta_proj.weight": (2, 32),
        "alpha.A_log": (2,), "alpha.dt_bias": (32,), "alpha.down.weight": (16, 32),
        "alpha.up.weight": (32, 16), "output_gate.down.weight": (16, 32),
        "output_gate.up.weight": (32, 16), "output_gate.up.bias": (32,),
        "softmax_gate.up.weight": (2, 32), "softmax_gate.up.bias": (2,), "norm.weight": (16,)}
    tensors = {}
    for block in range(2):
        for name, shape in shapes.items():
            tensors[f"{block}.{name}"] = torch.randn(shape).to(torch.bfloat16)
        tensors[f"{block}.beta_proj.weight"] = torch.randint(-30, 30, (2, 32), dtype=torch.int8)
        tensors[f"{block}.scale"] = torch.tensor(0.01)
    source = tmp_path / "original_branches.safetensors"
    save_file(tensors, str(source))
    weights = []
    for block in range(2):
        values = {}
        for name, shape in shapes.items():
            quant = {"format": "int8_tensorwise"} if name == "beta_proj.weight" else None
            values[name] = vdn_spec.LazyBranchTensor(str(source), f"{block}.{name}", torch.Size(shape),
                torch.bfloat16, f"{block}.scale" if quant else None, quant)
        weights.append(values)
    descriptor = dict(name="tiny-test", linear_head_dim=16, heads=2, head_dim=16)
    options = dict(NODE.SCHEMA["vdn"], retain_buffers="off", branch_weights="stream", short_conv=[])
    return NODE._restore_vdn(model, descriptor, weights, source.stat().st_size, options), source


def with_sla(model):
    return NODE._sla().patch_h3_sla(model, engine="comfy_kitchen", dense_backend="pytorch",
        sparsity_ratio=0.55, dense_steps="0,2", dense_last_steps=2, reference_protection="Light")


def hybrid_model(tmp_path):
    model, source = with_vdn(tmp_path, tiny_model())
    model = NODE._hybrid().VTS_H3HybridAttention().execute(model, dense_first_steps=0,
                                                        dense_last_steps=0, verbose=False)[0]
    return model, source


@pytest.mark.parametrize("variant", ["vdn", "hybrid", "sla"])
def test_native_h3_round_trip_and_patch_once(tmp_path, variant):
    model = tiny_model()
    if variant == "sla":
        model, source = with_sla(model), None
    else:
        model, source = with_vdn(tmp_path, model)
        if variant == "hybrid":
            model = NODE._hybrid().VTS_H3HybridAttention().execute(model, dense_first_steps=0,
                                                                dense_last_steps=0)[0]
    key = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    original = model.model.diffusion_model.blocks[0].attn.qkv_proj.weight.detach().clone()
    delta = torch.full_like(original, 0.125)
    model.add_patches({key: ("diff", (delta,))}, strength_patch=0.5)
    path = NODE._save(model, "roundtrip", sla_dense_backend="pytorch")
    if source:
        source.unlink()  # The bundle must not need the old branch file.
    loaded, settings = NODE._load(path)
    assert loaded.patches == {}
    torch.testing.assert_close(loaded.model.diffusion_model.blocks[0].attn.qkv_proj.weight,
                               original + delta * 0.5)
    assert json.loads(settings)["variant"] == variant
    saved, state = NODE._capture(loaded, sla_dense_backend="pytorch")
    if state:
        quant = state.branches[0].w["beta_proj.weight"].resolve(torch.device("cpu"))
        assert quant.dtype == torch.bfloat16
        assert state.branches[0].w["beta_proj.weight"]._path == path
    if variant == "hybrid":
        assert saved["runtime"]["hybrid"]["dense_first_steps"] == 0
        assert saved["runtime"]["hybrid"]["dense_last_steps"] == 0
    # Saving the loaded model again must not add the old adapter a second time.
    second = NODE._save(loaded, "resave", sla_dense_backend="pytorch")
    reloaded, _ = NODE._load(second)
    torch.testing.assert_close(reloaded.model.diffusion_model.blocks[0].attn.qkv_proj.weight,
                               original + delta * 0.5)


def test_runtime_zero_overrides_do_not_mutate_saved_recipe():
    runtime = deepcopy(NODE.SCHEMA)
    options = NODE.VTS_PreparedH3RuntimeOptions().execute(dense_first_steps=0, dense_last_steps=0,
        local_sparsity=0.0, retain_buffers="off",
        advanced_json='{"sampling":{"shift":9.0},"vdn":{"compile_scan":false}}')[0]
    result = NODE._apply_common(runtime, options)
    assert result["hybrid"]["dense_first_steps"] == result["hybrid"]["dense_last_steps"] == 0
    assert result["hybrid"]["local_sparsity"] == 0.0
    assert result["sampling"]["shift"] == 9.0
    assert result["vdn"]["retain_buffers"] == "off"
    assert runtime == NODE.SCHEMA


def test_saved_settings_are_default():
    assert NODE._apply_common(NODE.SCHEMA, NODE.VTS_PreparedH3RuntimeOptions().execute()[0]) == NODE.SCHEMA


@pytest.mark.parametrize("options", [
    {"vdn": {"strength": 0.5}}, {"vdn": {"linear_head_dim": 64}},
    {"sla": {"engine": "exec"}}, {"sampling": {"shift": "3"}},
    {"hybrid": {"dense_first_steps": False}}, {"vdn": {"compile_scan": 1}},
    {"vdn": {"short_conv": ["x"]}}, {"arbitrary_module": {}},
])
def test_unknown_weight_or_wrong_type_overrides_rejected(options):
    with pytest.raises(ValueError):
        NODE._validate_options(options)


def test_irrelevant_and_duplicate_overrides_fail():
    with pytest.raises(ValueError, match="does not apply"):
        NODE._apply_common({"vdn": NODE.SCHEMA["vdn"]}, {"common": {"dense_first_steps": 0}})
    with pytest.raises(ValueError, match="either"):
        NODE._apply_common(NODE.SCHEMA, {"common": {"dense_first_steps": 0}, "hybrid": {"dense_first_steps": 2}})


@pytest.mark.parametrize("prefix", ["../escape", "/tmp/escape", "a/b", "a\\b", "", "."])
def test_save_path_rejected_before_model_access(prefix):
    with pytest.raises(ValueError, match="prefix"):
        NODE._save(None, prefix)


def test_load_path_cannot_escape(tmp_path):
    outside = tmp_path / "outside.safetensors"
    save_file({"x": torch.zeros(1)}, str(outside))
    with pytest.raises(ValueError, match="inside"):
        NODE._resolve_file(str(outside))
    with pytest.raises(ValueError):
        NODE._resolve_file("../../outside.safetensors")


def test_unknown_runtime_patch_rejected(tmp_path):
    model, _ = with_vdn(tmp_path, tiny_model())
    model.add_wrapper_with_key("diffusion_model", "unknown", lambda *a: None)
    with pytest.raises(ValueError, match="Unsupported runtime wrapper"):
        NODE._capture(model)


def test_sla_backend_not_guessed_from_function():
    model = with_sla(tiny_model())
    with pytest.raises(ValueError, match="Cannot recover SLA"):
        NODE._capture(model)
    prompt = {"3": {"inputs": {"model": ["2", 0]}},
              "2": {"class_type": "H3SLAAttention", "inputs": {"dense_backend": "pytorch"}}}
    manifest, _ = NODE._capture(model, prompt=prompt, unique_id="3")
    assert manifest["runtime"]["sla"]["dense_backend"] == "pytorch"


def test_code_change_requires_explicit_opt_in(tmp_path, monkeypatch):
    model, _ = with_vdn(tmp_path, tiny_model())
    path = NODE._save(model, "compat")
    monkeypatch.setattr(NODE, "_fingerprints", lambda variant: {"changed": "yes"})
    with pytest.raises(ValueError, match="Attention code changed"):
        NODE._load(path)
    loaded, _ = NODE._load(path, compatibility="allow changed attention code")
    assert loaded.patches == {}


def test_partial_export_is_removed(tmp_path, monkeypatch):
    model, _ = with_vdn(tmp_path, tiny_model())
    def failing_writer(path, *args, **kwargs):
        save_file({"partial": torch.zeros(1)}, path)
        raise OSError("test write failure")
    monkeypatch.setattr(NODE, "_write_checkpoint", failing_writer)
    with pytest.raises(OSError, match="test write failure"):
        NODE._save(model, "failure")
    assert list((tmp_path / "models/diffusion_models/prepared-h3").iterdir()) == []


@pytest.mark.parametrize("quant_format", ["int8_tensorwise", "float8_e4m3fn"])
def test_native_quantized_patch_round_trip(tmp_path, quant_format):
    from comfy_kitchen.tensor import QuantizedTensor
    model, _ = with_vdn(tmp_path, tiny_model(quantized=quant_format))
    op = model.model.diffusion_model.blocks[0].attn.qkv_proj
    assert isinstance(op.weight, QuantizedTensor)
    key = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    delta = torch.full(op.weight.shape, 0.025)
    model.add_patches({key: ("diff", (delta,))}, strength_patch=0.5)
    expected = model.patch_weight_to_device(key, device_to=torch.device("cpu"), return_weight=True)
    assert isinstance(expected, QuantizedTensor)
    expected_sd = expected.state_dict(key)
    path = NODE._save(model, "quantized")
    with safe_open(path, framework="pt") as handle:
        raw_key = next(k for k in handle.keys() if k.endswith("blocks.0.attn.qkv_proj.weight"))
        assert handle.get_tensor(raw_key).dtype == (torch.int8 if quant_format == "int8_tensorwise" else torch.float8_e4m3fn)
    loaded, _ = NODE._load(path)
    actual = loaded.model.diffusion_model.blocks[0].attn.qkv_proj.weight
    assert isinstance(actual, QuantizedTensor)
    assert loaded.patches == {}
    for key, value in actual.state_dict(key).items():
        torch.testing.assert_close(value.reshape(-1).view(torch.uint8), expected_sd[key].reshape(-1).view(torch.uint8), rtol=0, atol=0)


def test_load_runtime_overrides_reach_actual_wrappers(tmp_path):
    model, _ = hybrid_model(tmp_path)
    path = NODE._save(model, "overrides")
    options = NODE.VTS_PreparedH3RuntimeOptions().execute(dense_first_steps=2, dense_last_steps=3,
        local_sparsity=0.6, retain_buffers="off", prefetch="off",
        advanced_json='{"sampling":{"shift":9.0,"audio_shift":2.0}}')[0]
    loaded, _ = NODE._load(path, options)
    manifest, state = NODE._capture(loaded)
    assert manifest["runtime"]["hybrid"]["dense_first_steps"] == 2
    assert manifest["runtime"]["hybrid"]["dense_last_steps"] == 3
    assert manifest["runtime"]["hybrid"]["local_sparsity"] == 0.6
    assert state.prefetch_mode == "off"
    assert not state.retain_buffers
    assert loaded.get_model_object("model_sampling").shift == 9.0
    assert loaded.get_model_object("model_sampling").audio_shift == 2.0
    assert loaded.patches == {}


def test_export_preserves_requested_auto_retain_policy(tmp_path):
    model, _ = with_vdn(tmp_path, tiny_model())
    prompt = {"3": {"inputs": {"model": ["2", 0]}},
              "2": {"class_type": "ApplyVDNH3", "inputs": {"retain_buffers": "auto"}}}
    manifest, _ = NODE._capture(model, prompt, "3")
    assert manifest["runtime"]["vdn"]["retain_buffers"] == "auto"


def test_unknown_attention_override_not_dropped(tmp_path):
    model, _ = with_vdn(tmp_path, tiny_model())
    model.model_options["transformer_options"]["optimized_attention_override"] = lambda *a: None
    with pytest.raises(ValueError, match="Unsupported attention override"):
        NODE._capture(model)


def test_sla_first_steps_override_reaches_wrapper(tmp_path):
    model = with_sla(tiny_model())
    path = NODE._save(model, "sla_steps", sla_dense_backend="pytorch")
    loaded, _ = NODE._load(path, {"common": {"dense_first_steps": 2, "dense_last_steps": 0},
                                 "sla": {"dense_steps": ""}})
    manifest, _ = NODE._capture(loaded)
    assert manifest["runtime"]["sla"]["dense_steps"] == "0,1"
    assert manifest["runtime"]["sla"]["dense_last_steps"] == 0


def test_custom_output_and_input_paths(tmp_path):
    model, _ = with_vdn(tmp_path, tiny_model())
    directory = tmp_path / "custom location" / "models"
    path = NODE.VTS_SavePreparedH3().execute(model, save_enabled=True, output_directory=str(directory))[0]
    assert Path(path).parent == directory
    assert Path(path).is_file()
    # The explicit input path is independent of the default dropdown roots.
    loaded, settings = NODE.VTS_LoadPreparedH3().execute("(no prepared files)", input_path=path)
    assert loaded.patches == {}
    assert json.loads(settings)["file"] == path
    connected, _ = NODE.VTS_LoadPreparedH3().execute("(no prepared files)", prepared_file=path)
    assert connected.patches == {}
    with pytest.raises(ValueError, match="either input_path"):
        NODE.VTS_LoadPreparedH3().execute("unused", input_path=path, prepared_file=path)


def test_custom_paths_require_absolute_paths():
    with pytest.raises(ValueError, match="absolute WSL input"):
        NODE._resolve_file("../file.safetensors", explicit=True)
    with pytest.raises(ValueError, match="absolute WSL output"):
        NODE._save(None, "valid", output_directory="../models")


def test_default_export_appears_in_standard_diffusion_picker(tmp_path):
    model, _ = with_vdn(tmp_path, tiny_model())
    path = NODE.VTS_SavePreparedH3().execute(model, save_enabled=True)[0]
    assert Path(path).parent == tmp_path / "models/diffusion_models/prepared-h3"
    name = "prepared-h3/" + Path(path).name
    choices = NODE.VTS_LoadPreparedH3.INPUT_TYPES()["required"]["prepared_model"][0]
    assert choices == folder_paths.get_filename_list("diffusion_models")
    assert name in choices
    loaded, _ = NODE.VTS_LoadPreparedH3().execute(name)
    assert not loaded.patches
    assert not (tmp_path / "output").exists()


def test_picker_uses_extra_diffusion_roots_and_rejects_unprepared_files(tmp_path):
    extra = tmp_path / "extra_models"
    extra.mkdir()
    path = extra / "ordinary.safetensors"
    save_file({"weight": torch.zeros(1)}, str(path))
    folder_paths.add_model_folder_path("diffusion_models", str(extra))
    choices = NODE.VTS_LoadPreparedH3.INPUT_TYPES()["required"]["prepared_model"][0]
    assert "ordinary.safetensors" in choices
    assert NODE._resolve_file("ordinary.safetensors") == path
    with pytest.raises(ValueError, match="Not a VTS prepared H3 file"):
        NODE.VTS_LoadPreparedH3().execute("ordinary.safetensors")


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float32])
@pytest.mark.parametrize("inference", [False, True])
def test_export_piece_is_gradient_free_and_does_not_copy(dtype, inference):
    with torch.inference_mode(inference):
        tensor = torch.zeros(4, dtype=dtype)
        piece = NODE._QuantizedExportPiece(None, "weight", tensor)
        assert not piece.requires_grad
        assert piece.dtype == dtype
        assert piece.data_ptr() == tensor.data_ptr()


@pytest.mark.parametrize("quant_format", ["int8_tensorwise", "float8_e4m3fn"])
def test_unmaterialized_quantized_export(tmp_path, monkeypatch, quant_format):
    """Exercise the lazy branch used by dynamic VRAM, even on a CPU test host."""
    from comfy_kitchen.tensor import QuantizedTensor
    model, _ = with_vdn(tmp_path, tiny_model(quantized=quant_format))
    key = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    op = model.model.diffusion_model.blocks[0].attn.qkv_proj
    assert not getattr(op, "comfy_patched_weights", False)
    original = op.weight
    model.add_patches({key: ("diff", (torch.full(op.weight.shape, 0.025),))}, strength_patch=0.5)
    expected = model.patch_weight_to_device(key, device_to=torch.device("cpu"), return_weight=True).state_dict(key)
    # Normal CPU loading eagerly patches weights; dynamic VRAM leaves them lazy.
    # Keep the actual ComfyUI ops unmaterialized to hit that same exporter path.
    monkeypatch.setattr(NODE.comfy.model_management, "load_models_gpu", lambda *a, **k: None)
    path = NODE._save(model, "lazy_quantized")
    assert op.weight is original  # Export must not replace live model weights.
    loaded, _ = NODE._load(path)
    weight = loaded.model.diffusion_model.blocks[0].attn.qkv_proj.weight
    assert isinstance(weight, QuantizedTensor)
    assert not loaded.patches
    for piece, value in weight.state_dict(key).items():
        torch.testing.assert_close(value.reshape(-1).view(torch.uint8),
            expected[piece].reshape(-1).view(torch.uint8), rtol=0, atol=0)
