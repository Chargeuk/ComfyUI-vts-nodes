

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
                "motion_frames": ("INT", {"default": 9, "min": -3, "step": 4, "tooltip": "The number of frames to process in a loop. Includes the motion frames."})
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0,
            frame_window_size=81, motion_frames=9):
        if motion_frames < 1:
            motion_frames = 0
        # increase the frame_window_size by 4 to account for dropping a frame each iteration
        # frame_window_size += 4

        # Extract the actual tensor from the latent dictionary
        latent_samples = latent_image["samples"]
        provided_number_of_latents = latent_samples.shape[2]
        provided_number_of_frames = int((provided_number_of_latents * 4) - 3)
        print(f"Provided latent shape: {latent_samples.shape}, provided_number_of_latents: {provided_number_of_latents}, provided_number_of_frames: {provided_number_of_frames}")

        batch_window_size = int((frame_window_size + 3) / 4)
        motion_sample_size = int((motion_frames + 3) / 4) if motion_frames > 0 else 0
        
        # Calculate step size (how many new frames to process each iteration)
        # Subtract 1 from step_size to account for dropping the last latent each loop
        base_step_size = batch_window_size - motion_sample_size
        step_size = batch_window_size - 1  # Drop last latent each loop
        number_of_loops = (provided_number_of_latents - motion_sample_size + step_size - 1) // step_size
        
        print(f"Number of loops: {number_of_loops}, Provided number of frames: {provided_number_of_frames}")
        print(f"Frame window size: {frame_window_size}, Motion frames: {motion_frames}")
        print(f"Batch window size: {batch_window_size}, Motion sample size: {motion_sample_size}, Step size: {step_size}")
        
        # Initialize samples tensor with None - will be set after first iteration
        samples = None

            # If we have fewer latents than the window size, process in single batch
        if provided_number_of_latents <= batch_window_size:
            print(f"Processing all {provided_number_of_latents} latents in single batch")
            return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

        for loop_idx in range(number_of_loops):
            # Calculate the starting index for new frames
            if loop_idx == 0:
                # First iteration: start from beginning of the provided latent samples
                start_idx = 0
                end_idx = min(batch_window_size, provided_number_of_latents)
                chunk_samples = latent_samples[:, :, start_idx:end_idx, :, :]
                total_chunk_samples = chunk_samples.shape[2]
                total_chunk_frames = int((total_chunk_samples * 4) - 3)
                print(f"first iteration, taking from latent_samples, start_idx={start_idx}, end_idx={end_idx} for {total_chunk_samples} samples = {total_chunk_frames} frames")
            else:
                # Subsequent iterations: combine motion frames + new frames
                # Start one latent earlier due to dropping last latent from previous batch
                # new_start_idx = motion_sample_size + (loop_idx - 1) * step_size
                number_of_latents_so_far = samples.shape[2]

                start_idx = number_of_latents_so_far
                end_idx = min(start_idx + base_step_size, provided_number_of_latents)
                
                # Get the last motion_sample_size frames from previous results
                motion_frames_tensor = samples[:, :, -motion_sample_size:, :, :]
                
                # Get new frames from the original latent
                new_frames = latent_samples[:, :, start_idx:end_idx, :, :]
                
                # Combine motion frames + new frames
                chunk_samples = torch.cat([motion_frames_tensor, new_frames], dim=2)
                total_chunk_samples = chunk_samples.shape[2]
                total_chunk_frames = int((total_chunk_samples * 4) - 3)
                print(f"Loop {loop_idx}, taking last {motion_sample_size} latents from samples & from latent_samples, start_idx={start_idx}, end_idx={end_idx} for {total_chunk_samples} samples = {total_chunk_frames} frames")

            chunk_number_of_frames = int((chunk_samples.shape[2] * 4) - 3)
            print(f"Loop {loop_idx}: Processing chunk with shape {chunk_samples.shape} which is {chunk_number_of_frames} frames")
            
            # Create a new latent dict for each chunk
            chunk_latent = latent_image.copy()
            chunk_latent["samples"] = chunk_samples
            
            batch_samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, chunk_latent, denoise=denoise)
            
            if loop_idx == 0:
                # First iteration: use all samples but drop the last latent
                samples = batch_samples #[:, :, :-1, :, :]  # Drop last latent
                number_of_samples = samples.shape[2]
                print(f"Loop {loop_idx}: Took {number_of_samples} samples from batch, dropped last latent, this gives {number_of_samples * 4} frames")
            else:
                # Subsequent iterations: blend the overlapping region and append new frames if motion_sample_size > 0
                if motion_sample_size > 0:
                    # Get the overlapping regions
                    existing_overlap = samples[:, :, -motion_sample_size:, :, :]  # Last motion_sample_size from existing
                    new_overlap = batch_samples[:, :, :motion_sample_size, :, :]  # First motion_sample_size from new batch
                    
                    # Create linear blending weights with extended range to avoid extremes
                    extended_size = motion_sample_size + 2  # Add 2 extra
                    extended_weights = torch.linspace(1.0, 0.0, extended_size, device=existing_overlap.device)
                    blend_weights = extended_weights[1:-1]  # Remove first and last elements
                    # Reshape weights to match tensor dimensions [1, 1, motion_sample_size, 1, 1]
                    blend_weights = blend_weights.view(1, 1, -1, 1, 1)
                    
                    # Perform linear blending
                    blended_overlap = existing_overlap * blend_weights + new_overlap * (1.0 - blend_weights)
                    blended_overlap_number_of_latents = blended_overlap.shape[2]
                    number_of_generated_latents = samples.shape[2]
                    print(f"Loop {loop_idx}: replacing {motion_sample_size} latents with {blended_overlap_number_of_latents} latents in current total of {number_of_generated_latents} latents")

                    # Replace the last motion_sample_size frames in samples with the blended version
                    samples[:, :, -motion_sample_size:, :, :] = blended_overlap
                    number_of_generated_latents = samples.shape[2]
                    print(f"Loop {loop_idx}: replaced {motion_sample_size} latents with {blended_overlap_number_of_latents} latents to give new total of {number_of_generated_latents} latents")

                # Append only the new frames (skip the overlapping region)
                new_samples_only = batch_samples[:, :, motion_sample_size:, :, :]
                
                # Drop the last latent except on the final loop
                # if loop_idx < number_of_loops - 1:  # Not the last loop
                #    new_samples_only = new_samples_only[:, :, :-1, :, :]  # Drop last latent

                new_samples_latent_count = new_samples_only.shape[2]

                if new_samples_latent_count > 0:  # Only concatenate if there are new frames
                    samples = torch.cat([samples, new_samples_only], dim=2)
                motion_sample_frames = motion_sample_size * 4
                new_samples_frames = new_samples_latent_count * 4
                total_added_frames = new_samples_frames + motion_sample_frames
                print(f"Loop {loop_idx}: Blended {motion_sample_size} overlapping latents={motion_sample_frames} frames, added {new_samples_latent_count} new latents={new_samples_frames} for a total of {total_added_frames} frames")
                number_of_generated_latents = samples.shape[2]
                number_of_generated_frames = int((number_of_generated_latents * 4) - 3)
                print(f"Loop {loop_idx} samples shape: {samples.shape}, which is {number_of_generated_latents} latents, and {number_of_generated_frames} frames")

        number_of_generated_latents = samples.shape[2]
        number_of_generated_frames = int((number_of_generated_latents * 4) - 3)
        print(f"Final samples shape: {samples.shape}, which is {number_of_generated_latents} latents, and {number_of_generated_frames} frames")
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
