# VTS KSampler (Advanced)

Search for **VTS KSampler (Advanced)** in `VTS/sampling`.

Uses ComfyUI's advanced sampling controls with the regular VTS KSampler's per-image seed modes, conditioning-list support, and sequential latent-batch processing. The longest positive list, negative list, or latent batch determines the output count. Shorter inputs repeat their last item. Outputs are combined into one LATENT batch.

- `noise_seed`: base seed. `seed_per_image` can keep it fixed, increment it, decrement it, or derive reproducible random seeds from it.
- `add_noise`: enable for an initial sampling pass; disable when continuing a latent that already contains noise.
- `start_at_step` / `end_at_step`: select the part of the sampling schedule to run.
- `return_with_leftover_noise`: enable to keep remaining noise for another sampler; disable to force full denoising at the end of the selected range.

For example, with 20 total steps, run steps 0–10 with noise enabled and leftover noise enabled. Feed the latent into another advanced sampler with the same schedule, steps 10–20, noise disabled, and leftover noise disabled.

As in the built-in advanced node, the step range controls partial sampling rather than a `denoise` widget. The regular VTS KSampler is unchanged and retains that widget. Model, step count, CFG, sampler, scheduler, and advanced controls are shared across all outputs. A video latent is split along its batch dimension, preserving the frames within each video.
