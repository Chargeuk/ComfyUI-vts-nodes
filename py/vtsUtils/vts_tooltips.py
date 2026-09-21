"""Schema-only help for VTS controls and sockets; never changes execution."""
import copy
import functools

DEVICE_HELP = (
    "Used only when latent_device_policy (or vts_latent_device_policy) is Specified. "
    "cpu loads into system RAM; cuda:0 loads onto the first visible NVIDIA GPU; "
    "cuda:1 requires a second visible GPU; cuda uses the current CUDA device. "
    "This places tensors loaded from DiskLatent, not the model. Existing in-memory "
    "tensors are unchanged. The selected device must be available and have enough memory."
)
POLICY_HELP = (
    "Where DiskLatent tensors are loaded before processing: Original restores each "
    "tensor's saved device (or its deferred device override); CPU uses system RAM; "
    "Specified uses the device text field. Existing in-memory tensors are unchanged."
)
IMAGE_CONTROLS = {
    "prefix": "Filename prefix for DiskImage output. Defaults to the node's naming scheme.",
    "output_dir": "Folder for DiskImage files. Relative paths are resolved from ComfyUI's working directory. Used when saving disk output.",
    "start_sequence": "Starting number for output filenames. List-mapped calls include their list position to distinguish outputs.",
    "format": "DiskImage file format: PNG is lossless; JPEG is lossy; WebP supports lossy or lossless mode. Does not affect DiskLatent files.",
    "num_workers": "Number of workers used to write image files. More workers can improve batch saving but use more resources.",
    "compression_level": "Image compression effort: PNG 0-9; WebP 0-6. Higher settings take longer. Does not control latent compression.",
    "quality": "JPEG/WebP quality from 1-100. For WebP, 101 selects lossless mode. PNG ignores this setting; JPEG cannot be lossless.",
}
TYPE_HELP = {
    "IMAGE": "Image pixels, normally a tensor shaped [batch, height, width, channels].",
    "LATENT": "Latent dictionary containing samples and optional masks or other metadata. Image samples are usually [batch, channels, height, width]; video/audio can use other layouts.",
    "MASK": "Mask tensor: usually 0 excludes a pixel and 1 includes it; intermediate values give partial coverage.",
    "MODEL": "Diffusion model used by the node.",
    "VAE": "VAE used to encode pixels into latents or decode latents into pixels.",
    "CONDITIONING": "Conditioning data, such as encoded prompts and associated metadata.",
}


def _join(existing, extra):
    if not extra or extra in (existing or ""):
        return existing
    return ((existing + " ") if existing else "") + extra


def socket_help(kind, output=False, disk_latent=False, disk_image=False):
    text = TYPE_HELP.get(kind, "") if isinstance(kind, str) else ""
    if kind == "LATENT" and disk_latent:
        text += (" May be native tensors or a DiskLatent file reference, selected by the latent return-type control. DiskLatent metadata exposes shape, dtype and device without loading tensor data." if output else
                 " Accepts native latents or DiskLatent. Disk-backed tensors are loaded before computation using the latent device policy.")
    if kind == "IMAGE" and disk_image:
        text += (" Returns a tensor or DiskImage file reference according to the image return-type control." if output else
                 " Also accepts DiskImage; pixel data is loaded when needed for processing.")
    return text


def _control_help(name, image_controls):
    if name in ("latent_device", "vts_latent_device"):
        return DEVICE_HELP
    if name in ("latent_device_policy", "vts_latent_device_policy"):
        return POLICY_HELP
    key = name[4:] if name.startswith("vts_") else name
    return IMAGE_CONTROLS.get(key) if image_controls else None


def document_node(cls, *, disk_latent=False, disk_image=False):
    """Preserve existing help and add descriptions to both legacy and V3 schemas."""
    if cls.__dict__.get("_vts_documented", False):
        return cls
    from comfy_api.latest import io
    if issubclass(cls, io.ComfyNode):
        original = cls.define_schema

        @classmethod
        def define_schema(current_cls):
            schema = copy.deepcopy(original.__func__(current_cls))
            image_controls = any(item.id in ("return_type", "vts_return_type") for item in schema.inputs)

            def annotate(item, output=False):
                kind = item.get_io_type()
                extra = socket_help(kind, output, disk_latent, disk_image)
                control = _control_help(item.id, image_controls) if not output else None
                item.tooltip = control or _join(getattr(item, "tooltip", None), extra)
                template = getattr(item, "template", None)
                child = getattr(template, "input", None)
                if child is not None:
                    annotate(child)
            for item in schema.inputs:
                annotate(item)
            for item in schema.outputs:
                annotate(item, True)
            return schema
        cls.define_schema = define_schema
    else:
        original = cls.INPUT_TYPES

        @classmethod
        @functools.wraps(original)
        def input_types(current_cls):
            schema = copy.deepcopy(original.__func__(current_cls))
            names = set(schema.get("required", {})) | set(schema.get("optional", {}))
            image_controls = bool(names & {"return_type", "vts_return_type"})
            for group in ("required", "optional"):
                for name, spec in schema.get(group, {}).items():
                    config = dict(spec[1]) if len(spec) > 1 else {}
                    extra = socket_help(spec[0], False, disk_latent, disk_image)
                    tooltip = _control_help(name, image_controls) or _join(config.get("tooltip"), extra)
                    if tooltip:
                        config["tooltip"] = tooltip
                        schema[group][name] = (spec[0], config, *spec[2:])
            return schema
        cls.INPUT_TYPES = input_types
        old = getattr(cls, "OUTPUT_TOOLTIPS", ()) or ()
        cls.OUTPUT_TOOLTIPS = tuple(_join(old[i] if i < len(old) else None,
            socket_help(kind, True, disk_latent, disk_image)) or ""
            for i, kind in enumerate(getattr(cls, "RETURN_TYPES", ())))
    cls._vts_documented = True
    return cls
