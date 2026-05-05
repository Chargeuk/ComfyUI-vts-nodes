import os

import folder_paths
import torch
from comfy import model_management
from comfy.sd import VAE
from comfy.utils import load_torch_file


class VTS_VAELoader:
    video_taes = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5"]
    image_taes = ["taesd", "taesdxl", "taesd3", "taef1"]

    @staticmethod
    def vae_list(s):
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
            elif v.startswith("taef1_encoder."):
                f1_taesd_dec = True
            elif v.startswith("taef1_decoder."):
                f1_taesd_enc = True
            else:
                for tae in s.video_taes:
                    if v.startswith(tae):
                        vaes.append(v)

        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append("taef1")
        vaes.append("pixel_space")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith(f"{name}_encoder."), approx_vaes))
        decoder = next(filter(lambda a: a.startswith(f"{name}_decoder."), approx_vaes))

        enc = load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc:
            sd[f"taesd_encoder.{k}"] = enc[k]

        dec = load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec:
            sd[f"taesd_decoder.{k}"] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (s.vae_list(s),),
                "device": (["main_device", "cpu"],),
                "weight_dtype": (["bf16", "fp16", "fp32"],),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "VTS/vae"

    def load_vae(self, vae_name, device, weight_dtype):
        metadata = None
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[weight_dtype]
        if device == "main_device":
            device = model_management.get_torch_device()
        elif device == "cpu":
            device = torch.device("cpu")

        if vae_name == "pixel_space":
            sd = {"pixel_space_vae": torch.tensor(1.0)}
        elif vae_name in self.image_taes:
            sd = self.load_taesd(vae_name)
        else:
            if os.path.splitext(vae_name)[0] in self.video_taes:
                vae_path = folder_paths.get_full_path_or_raise("vae_approx", vae_name)
            else:
                vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd, metadata = load_torch_file(vae_path, return_metadata=True)

        is_audio_vae = (
            "vocoder.conv_post.weight" in sd
            or "vocoder.vocoder.conv_post.weight" in sd
            or "vocoder.resblocks.0.convs1.0.weight" in sd
            or "vocoder.vocoder.resblocks.0.convs1.0.weight" in sd
        )

        if is_audio_vae:
            import comfy.utils

            sd_audio = comfy.utils.state_dict_prefix_replace(
                dict(sd), {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."}, filter_keys=True
            )
            vae = VAE(sd=sd_audio, device=device, dtype=dtype, metadata=metadata)
        else:
            vae = VAE(sd=sd, device=device, dtype=dtype, metadata=metadata)

        vae.throw_exception_if_invalid()
        return (vae,)


NODE_CLASS_MAPPINGS = {
    "VAELoader VTS": VTS_VAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAELoader VTS": "VAELoader VTS",
}
