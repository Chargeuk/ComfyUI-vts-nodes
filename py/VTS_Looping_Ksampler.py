

import logging
from spandrel import ModelLoader, ImageModelDescriptor
from spandrel.__helpers.size_req import pad_tensor
from comfy import model_management
import torch
import comfy
import comfy.utils
import folder_paths 


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=None, disable_pbar=disable_pbar, seed=seed)
    return samples
    # out = latent.copy()
    # out["samples"] = samples
    # return (out, )

class VTSLoopingKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            },
            "optional": {
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for encoding and decoding the provided images."}),
                "frame_window_size": ("INT", {"default": 81, "min": 1, "step": 4, "tooltip": "The number of frames to process in a loop. Includes the motion frames."}),
                "motion_frames": ("INT", {"default": 9, "min": 1, "step": 4, "tooltip": "The number of frames to process in a loop. Includes the motion frames."})
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0,
            frame_window_size=81, motion_frames=9):
        
        # Extract the actual tensor from the latent dictionary
        latent_samples = latent_image["samples"]
        print(f"Provided latent shape: {latent_samples.shape}")

        batch_window_size = int((frame_window_size + 3) / 4)

        provided_number_of_latents = latent_samples.shape[2]
        provided_number_of_frames = int((provided_number_of_latents * 4) - 3)
        number_of_loops = (provided_number_of_frames + frame_window_size - 1) // frame_window_size
        print(f"Number of loops: {number_of_loops}, Provided number of frames: {provided_number_of_frames}, Frame window size: {frame_window_size}")
        
        # Initialize samples tensor with None - will be set after first iteration
        samples = None
        
        for i in range(0, provided_number_of_latents, batch_window_size):
            print(f"Processing chunk starting at index {i}")
            # Create a new latent dict for each chunk
            chunk_latent = latent_image.copy()
            chunk_latent["samples"] = latent_samples[:, :, i:i + batch_window_size, :, :]
            
            batch_samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, chunk_latent, denoise=denoise)
            
            # Append to the samples tensor - concatenate along dimension 2 (the batch dimension)
            if samples is None:
                samples = batch_samples
            else:
                samples = torch.cat([samples, batch_samples], dim=2)

        print(f"Final samples shape: {samples.shape}")
        out = latent_image.copy()
        out["samples"] = samples
        return (out, )

    
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Looping K Sampler": VTSLoopingKSampler
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Looping K Sampler": "Looping K Sampler"
}
