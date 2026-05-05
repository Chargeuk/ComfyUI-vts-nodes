import importlib
import importlib.util
import inspect
import os
import sys

import torch
from comfy import model_management
from comfy_api.latest import io

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import DiskImage


def _load_upstream_iclora_module():
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if "ltx" not in name.lower():
            continue
        candidate = getattr(module, "LTXAddVideoICLoRAGuide", None)
        if inspect.isclass(candidate):
            return module

    package_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "ComfyUI-LTXVideo")
    )
    init_path = os.path.join(package_dir, "__init__.py")
    if not os.path.exists(init_path):
        raise ImportError(
            f"ComfyUI-LTXVideo not found at expected path: {package_dir}"
        )

    package_name = "_vts_upstream_comfyui_ltxvideo"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_path,
            submodule_search_locations=[package_dir],
        )
        if spec is None or spec.loader is None:
            raise ImportError("Unable to create import spec for ComfyUI-LTXVideo")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)

    module = importlib.import_module(f"{package_name}.iclora")
    candidate = getattr(module, "LTXAddVideoICLoRAGuide", None)
    if not inspect.isclass(candidate):
        raise ImportError("ComfyUI-LTXVideo iclora module did not expose LTXAddVideoICLoRAGuide")
    return module


_UPSTREAM_ICLORA_MODULE = _load_upstream_iclora_module()
_UPSTREAM_GUIDE_CLASS = _UPSTREAM_ICLORA_MODULE.LTXAddVideoICLoRAGuide


class VTS_LTXAddVideoICLoRAGuide(_UPSTREAM_GUIDE_CLASS):
    @classmethod
    def define_schema(cls):
        schema = _UPSTREAM_GUIDE_CLASS.define_schema()
        schema.node_id = "VTSLTXAddVideoICLoRAGuide"
        schema.display_name = "VTS Add Video IC-LoRA Guide"
        schema.category = "VTS/Lightricks/IC-LoRA"
        description = schema.description or ""
        extra = " Supports DiskImage inputs for the image input by materializing them before delegating to the upstream LTX node."
        if extra.strip() not in description:
            schema.description = f"{description}{extra}".strip()
        return schema

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        image = kwargs.get("image")
        materialized_image = None

        if isinstance(image, DiskImage):
            kwargs = dict(kwargs)
            materialized_image = image.materialize()
            kwargs["image"] = materialized_image
        elif image is not None and not isinstance(image, torch.Tensor):
            raise TypeError(
                "VTS Add Video IC-LoRA Guide expected the image input to be a tensor or DiskImage."
            )

        try:
            return super().execute(**kwargs)
        finally:
            if materialized_image is not None:
                del materialized_image
                model_management.soft_empty_cache()


NODE_CLASS_MAPPINGS = {
    "VTSLTXAddVideoICLoRAGuide": VTS_LTXAddVideoICLoRAGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTSLTXAddVideoICLoRAGuide": "VTS Add Video IC-LoRA Guide",
}
