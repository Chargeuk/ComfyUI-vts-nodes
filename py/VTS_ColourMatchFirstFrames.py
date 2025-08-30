import comfy
import comfy.utils
import node_helpers
import torch
from nodes import MAX_RESOLUTION

def colormatch(image_ref, image_target, method, strength=1.0, editInPlace=False, gc_interval=50):
    try:
        from color_matcher import ColorMatcher
    except ImportError:
        raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")
    
    # Early validation
    if image_ref.dim() != 4 or image_target.dim() != 4:
        raise ValueError("ColorMatch: Expected 4D tensors (batch, height, width, channels)")
    
    batch_size = image_target.size(0)
    ref_batch_size = image_ref.size(0)
    
    # Validate batch sizes early
    if ref_batch_size > 1 and ref_batch_size != batch_size:
        raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")
    
    # Move to CPU efficiently (avoid redundant moves)
    if image_ref.device != torch.device('cpu'):
        image_ref = image_ref.cpu()
    if image_target.device != torch.device('cpu'):
        image_target = image_target.cpu()
    
    # Handle output tensor allocation
    if editInPlace:
        out = image_target
    else:
        out = torch.empty_like(image_target, dtype=torch.float32, device='cpu')
    
    # Initialize ColorMatcher once
    cm = ColorMatcher()
    
    # Process each image in the batch
    for i in range(batch_size):
        # Get individual images (avoid squeeze - use direct indexing)
        target_img = image_target[i]  # Shape: [H, W, C]
        ref_img = image_ref[0] if ref_batch_size == 1 else image_ref[i]  # Shape: [H, W, C]
        
        # Convert to numpy only when needed
        target_np = target_img.numpy()
        ref_np = ref_img.numpy()
        
        try:
            # Perform color matching
            result_np = cm.transfer(src=target_np, ref=ref_np, method=method)
            
            # Apply strength multiplier efficiently
            if strength != 1.0:
                result_np = target_np + strength * (result_np - target_np)
            
            # Convert back to tensor and update output
            result_tensor = torch.from_numpy(result_np)
            
            if editInPlace:
                image_target[i].copy_(result_tensor)
            else:
                out[i].copy_(result_tensor)
            
            # Clean up intermediate variables
            del target_np, ref_np, result_np, result_tensor
            
            # Garbage collection at intervals
            if gc_interval > 0 and (i + 1) % gc_interval == 0:
                import gc
                gc.collect()
                
        except Exception as e:
            print(f"Error occurred during transfer for image {i}: {e}")
            # Continue processing other images rather than breaking
            continue
    
    # Ensure output is float32 and properly clamped
    if not editInPlace and out.dtype != torch.float32:
        out = out.to(torch.float32)
    out.clamp_(0, 1)
    
    return (out,)


class VTS_ColourMatchFirstFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_target": ("IMAGE",),
                "method": (
            [   
                'mkl',
                'hm', 
                'reinhard', 
                'mvgd', 
                'hm-mvgd-hm', 
                'hm-mkl-hm',
            ], {
               "default": 'mkl'
            }),
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "When true, bypass processing and return images unchanged"}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "numberOfFirstFrames": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1, "tooltip": "Number of first frames to use as brightness reference library"}),
                "contrast_stabilization": ("BOOLEAN", {"default": False, "tooltip": "Enable contrast stabilization to prevent shadow/highlight drift"}),
                "shadow_threshold": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.5, "step": 0.05, "tooltip": "Luminance threshold to define shadow areas"}),
                "highlight_threshold": ("FLOAT", {"default": 0.7, "min": 0.5, "max": 0.9, "step": 0.05, "tooltip": "Luminance threshold to define highlight areas"}),
                "shadow_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Strength of shadow correction - prevents shadows from becoming darker and less saturated. 0.0=no correction, 1.0=full correction, >1.0=over-correction"}),
                "highlight_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Strength of highlight correction - prevents highlights from becoming lighter and less saturated. 0.0=no correction, 1.0=full correction, >1.0=over-correction"}),
                "shadow_anti_banding": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Anti-banding smoothing for shadows. Higher values = smoother shadows but potentially softer detail"}),
                "highlight_anti_banding": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Anti-banding smoothing for highlights. Higher values = smoother highlights but potentially softer detail"}),
                "editInPlace": ("BOOLEAN", {"default": False, "tooltip": "When true, modify the input image_target tensor directly instead of creating a new tensor"}),
                "gc_interval": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1, "tooltip": "Garbage collection interval. Set to 0 to disable automatic garbage collection. For large batches, lower values can help manage memory"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    DESCRIPTION = """
First Frames Histogram Color Matching with Contrast Stabilization

This node implements a simple and effective color matching strategy that uses the first N frames 
of a video sequence as reference frames. It builds a brightness signature library from these 
initial frames and matches subsequent frames to the most similar reference frame for color correction.

Key Features:
- Clean, focused implementation without adaptive complexity
- Histogram-based brightness matching for robust frame selection
- Consistent color correction using actual reference frames
- Optional contrast stabilization to prevent shadow/highlight drift
- Separate strength controls for shadows and highlights
- Advanced anti-banding with smooth transitions and local corrections
- Addresses temporal contrast expansion and shadow desaturation
- Ideal for maintaining temporal color and tonal stability in video sequences

The first frames are left unchanged and used to build the reference library. 
All subsequent frames are color-matched to the best-fitting reference frame.

Contrast Stabilization (Optional):
- Prevents shadows from getting darker and losing saturation over time
- Prevents highlights from becoming blown out and losing detail
- Uses statistical analysis to maintain tonal stability
- Individual strength controls for shadow_strength and highlight_strength
- Advanced anti-banding with smooth zone transitions
- Edge-preserving smoothing to eliminate banding artifacts while preserving detail
"""

    CATEGORY = "VTS"

    def colormatch(self, image_target, method, passthrough, strength=1.0, numberOfFirstFrames=20, 
                   contrast_stabilization=False, shadow_threshold=0.3, highlight_threshold=0.7, 
                   shadow_strength=0.8, highlight_strength=0.8, shadow_anti_banding=0.3, 
                   highlight_anti_banding=0.2, editInPlace=False, gc_interval=50):
        if passthrough:
            print("VTS_ColourMatchFirstFrames - passthrough is True, returning original image_target without processing")
            return (image_target,)
        
        mode_desc = "color matching"
        if contrast_stabilization:
            mode_desc += " with contrast stabilization"
        
        print(f"VTS_ColourMatchFirstFrames - Processing {image_target.shape[0]} frames with first {numberOfFirstFrames} as reference library ({mode_desc})")
        
        # Initialize brightness lookup
        self.brightness_lookup = []
        self.numberOfFirstFrames = numberOfFirstFrames
        
        # Process the sequence
        output = self.process_first_frames_sequence(
            image_target,
            method,
            strength,
            editInPlace,
            gc_interval,
            contrast_stabilization,
            shadow_threshold,
            highlight_threshold,
            shadow_strength,
            highlight_strength,
            shadow_anti_banding,
            highlight_anti_banding
        )
        
        print(f"VTS_ColourMatchFirstFrames - Finished processing. Built reference library with {len(self.brightness_lookup)} frames")
        return (output,)
    
    def calculate_brightness_signature(self, frame):
        """Calculate comprehensive brightness signature for frame matching"""
        lum = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        
        # Histogram-based signature (most important)
        hist = torch.histc(lum.flatten(), bins=16, min=0, max=1)
        hist_norm = hist / (hist.sum() + 1e-8)
        
        # Statistical measures
        mean_lum = lum.mean().item()
        std_lum = lum.std().item()
        
        # Percentiles for distribution shape
        percentiles = torch.quantile(lum.flatten(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        
        return {
            'histogram': hist_norm,
            'mean': mean_lum,
            'std': std_lum,
            'percentiles': percentiles
        }

    def find_best_brightness_match(self, target_signature, brightness_lookup):
        """Find the best matching frame from brightness lookup"""
        best_distance = float('inf')
        best_frame_data = None
        
        for frame_data in brightness_lookup:
            ref_sig = frame_data['signature']
            
            # Primary: Histogram distance
            hist_distance = torch.sum(torch.abs(target_signature['histogram'] - ref_sig['histogram'])).item()
            
            # Secondary: Statistical distances
            mean_distance = abs(target_signature['mean'] - ref_sig['mean'])
            std_distance = abs(target_signature['std'] - ref_sig['std'])
            perc_distance = torch.mean(torch.abs(target_signature['percentiles'] - ref_sig['percentiles'])).item()
            
            # Weighted combination
            total_distance = (0.6 * hist_distance + 
                             0.2 * mean_distance + 
                             0.1 * std_distance + 
                             0.1 * perc_distance)
            
            if total_distance < best_distance:
                best_distance = total_distance
                best_frame_data = frame_data
        
        return best_frame_data

    def apply_contrast_stabilization(self, current_frame, shadow_strength=0.8, highlight_strength=0.8, 
                                   shadow_threshold=0.3, highlight_threshold=0.7, 
                                   shadow_anti_banding=0.3, highlight_anti_banding=0.2):
        """Apply contrast stabilization to prevent shadow/highlight drift without using reference frames"""
        curr_lum = 0.299 * current_frame[..., 0] + 0.587 * current_frame[..., 1] + 0.114 * current_frame[..., 2]
        
        result = current_frame.clone()
        
        # Create smooth transition masks instead of hard thresholds
        # Shadow mask: strong at 0, fades to 0 at shadow_threshold, extends slightly beyond
        shadow_fade_end = shadow_threshold + 0.1
        shadow_mask = torch.clamp((shadow_fade_end - curr_lum) / (shadow_fade_end - 0.0), 0.0, 1.0)
        shadow_mask = shadow_mask * (curr_lum < shadow_fade_end).float()
        
        # Highlight mask: starts at highlight_threshold, strong at 1.0, with smooth transition
        highlight_fade_start = highlight_threshold - 0.1
        highlight_mask = torch.clamp((curr_lum - highlight_fade_start) / (1.0 - highlight_fade_start), 0.0, 1.0)
        highlight_mask = highlight_mask * (curr_lum > highlight_fade_start).float()
        
        # Shadow stabilization: prevent shadows from getting darker and less saturated
        if shadow_mask.sum() > 0 and shadow_strength > 0:
            # Calculate per-pixel lift factor based on how dark the pixel is
            lift_factor = torch.ones_like(curr_lum)
            shadow_areas = shadow_mask > 0.1
            
            if shadow_areas.any():
                # Stronger lift for darker pixels, gentler for lighter shadows
                darkness_factor = (shadow_threshold - curr_lum[shadow_areas]) / shadow_threshold
                lift_amount = 1.0 + (darkness_factor * 0.2 * shadow_strength)  # Max 20% lift
                lift_factor[shadow_areas] = lift_amount
            
            # Apply luminance lift and saturation restoration
            for c in range(3):
                # Lift shadows
                shadow_correction = (current_frame[..., c] * lift_factor - current_frame[..., c]) * shadow_mask
                
                # Restore saturation by pushing towards channel mean in shadow areas
                if shadow_areas.any():
                    channel_mean = current_frame[..., c][shadow_mask > 0.1].mean()
                    saturation_restore = (channel_mean - current_frame[..., c]) * 0.1 * shadow_strength * shadow_mask
                    result[..., c] = result[..., c] + shadow_correction + saturation_restore
                else:
                    result[..., c] = result[..., c] + shadow_correction
        
        # Highlight stabilization: prevent highlights from getting lighter and blown out
        if highlight_mask.sum() > 0 and highlight_strength > 0:
            # Calculate per-pixel compression factor based on how bright the pixel is
            compress_factor = torch.ones_like(curr_lum)
            highlight_areas = highlight_mask > 0.1
            
            if highlight_areas.any():
                # Stronger compression for brighter pixels
                brightness_factor = (curr_lum[highlight_areas] - highlight_threshold) / (1.0 - highlight_threshold)
                compress_amount = 1.0 - (brightness_factor * 0.15 * highlight_strength)  # Max 15% compression
                compress_factor[highlight_areas] = torch.clamp(compress_amount, 0.7, 1.0)
            
            # Apply luminance compression and detail restoration
            for c in range(3):
                # Compress highlights
                highlight_correction = (current_frame[..., c] * compress_factor - current_frame[..., c]) * highlight_mask
                
                # Restore detail by preserving local contrast
                if highlight_areas.any():
                    local_mean = torch.nn.functional.avg_pool2d(
                        current_frame[..., c].unsqueeze(0).unsqueeze(0), 
                        kernel_size=5, stride=1, padding=2
                    ).squeeze()
                    
                    # Make detail preservation proportional to compression
                    # Get average compression factor for highlight areas
                    avg_compress_factor = compress_factor[highlight_areas].mean().item()
                    
                    # Scale detail preservation inversely to compression
                    # When compress_factor = 1.0 (no compression) → detail_strength = 0.8
                    # When compress_factor = 0.7 (max compression) → detail_strength ≈ 0.56
                    detail_strength = 0.8 * avg_compress_factor
                    
                    detail_preserve = (current_frame[..., c] - local_mean) * detail_strength * highlight_mask
                    result[..., c] = result[..., c] + highlight_correction + detail_preserve
                else:
                    result[..., c] = result[..., c] + highlight_correction
        
        # Apply anti-banding smoothing
        if shadow_anti_banding > 0 and shadow_mask.sum() > 0:
            result = self.apply_zone_smoothing(result, current_frame, shadow_mask, shadow_anti_banding, "shadow")
        
        if highlight_anti_banding > 0 and highlight_mask.sum() > 0:
            result = self.apply_zone_smoothing(result, current_frame, highlight_mask, highlight_anti_banding, "highlight")
        
        result = torch.clamp(result, 0.0, 1.0)
        return result

    def apply_zone_smoothing(self, result, original, mask, smoothing_strength, zone_type):
        """Fast edge-preserving smoothing with optimized operations"""
        if smoothing_strength <= 0:
            return result
        
        # Reduced kernel sizes for speed
        kernel_size = 3  # Same size for both shadow and highlight for simplicity
        padding = kernel_size // 2
        
        # Simple edge detection (faster than full gradient calculation)
        orig_gray = 0.299 * original[..., 0] + 0.587 * original[..., 1] + 0.114 * original[..., 2]
        
        # Faster edge detection using built-in conv2d
        edge_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                   dtype=torch.float32, device=original.device).unsqueeze(0).unsqueeze(0)
        
        orig_padded = orig_gray.unsqueeze(0).unsqueeze(0)
        orig_padded = torch.nn.functional.pad(orig_padded, (1, 1, 1, 1), mode='reflect')
        edge_response = torch.nn.functional.conv2d(orig_padded, edge_kernel).squeeze()
        edge_strength = torch.abs(edge_response)
        
        # Simplified edge weighting
        edge_threshold = 0.1
        smooth_weights = torch.exp(-edge_strength / edge_threshold)
        zone_smooth_weights = mask * smooth_weights * smoothing_strength
        
        # Fast box filter using separable convolution
        smoothed = result.clone()
        box_kernel = torch.ones(1, 1, kernel_size, 1, device=result.device) / kernel_size
        
        for c in range(3):
            channel = result[..., c].unsqueeze(0).unsqueeze(0)
            # Horizontal pass
            channel_h = torch.nn.functional.pad(channel, (0, 0, padding, padding), mode='reflect')
            channel_h = torch.nn.functional.conv2d(channel_h, box_kernel)
            # Vertical pass  
            channel_v = torch.nn.functional.pad(channel_h, (padding, padding, 0, 0), mode='reflect')
            channel_smooth = torch.nn.functional.conv2d(channel_v, box_kernel.transpose(-1, -2))
            
            # Blend with original
            blend_factor = zone_smooth_weights
            smoothed[..., c] = (1 - blend_factor) * result[..., c] + blend_factor * channel_smooth.squeeze()
        
        return smoothed

    def process_first_frames_sequence(self, decoded_frames, color_match_method, color_match_strength, editInPlace, gc_interval,
                                     contrast_stabilization=False, shadow_threshold=0.3, highlight_threshold=0.7, 
                                     shadow_strength=0.8, highlight_strength=0.8, shadow_anti_banding=0.3, highlight_anti_banding=0.2):
        """
        Enhanced processing with optional contrast stabilization
        """
        if color_match_strength <= 0.0 and not contrast_stabilization:
            return decoded_frames
        
        processed_frames = []
        
        for i, current_frame in enumerate(decoded_frames):
            current_frame_batch = current_frame.unsqueeze(0)  # Add batch dimension
            
            if self.brightness_lookup is None or len(self.brightness_lookup) < self.numberOfFirstFrames:
                # Build brightness lookup from first frames (no color matching applied)
                signature = self.calculate_brightness_signature(current_frame)
                self.brightness_lookup.append({
                    'frame': current_frame.clone(),
                    'signature': signature,
                    'frame_index': i
                })
                print(f"Frame {i}: Added to brightness lookup (mean luminance: {signature['mean']:.3f}, std: {signature['std']:.3f})")
                
                # Apply contrast stabilization to reference frames if enabled
                if contrast_stabilization:
                    processed_frame = self.apply_contrast_stabilization(
                        current_frame, shadow_strength, highlight_strength,
                        shadow_threshold, highlight_threshold,
                        shadow_anti_banding, highlight_anti_banding
                    )
                else:
                    processed_frame = current_frame
                    
                processed_frames.append(processed_frame)
            else:
                # Use brightness lookup to find best matching reference frame
                target_signature = self.calculate_brightness_signature(current_frame)
                best_match = self.find_best_brightness_match(target_signature, self.brightness_lookup)
                
                if best_match is not None:
                    reference_frame = best_match['frame'].unsqueeze(0)
                    print(f"Frame {i}: Matched with reference frame {best_match['frame_index']} (target mean lum: {target_signature['mean']:.3f})")
                    
                    # Apply color matching using the matched reference frame
                    if color_match_strength > 0.0:
                        color_match_result = colormatch(reference_frame, current_frame_batch, color_match_method, color_match_strength, editInPlace, gc_interval)
                        processed_frame = color_match_result[0][0]  # Remove batch dimension
                    else:
                        processed_frame = current_frame
                else:
                    print(f"Frame {i}: No suitable match found in brightness lookup, using original frame")
                    processed_frame = current_frame
                
                # Apply contrast stabilization if enabled
                if contrast_stabilization:
                    processed_frame = self.apply_contrast_stabilization(
                        processed_frame, shadow_strength, highlight_strength,
                        shadow_threshold, highlight_threshold,
                        shadow_anti_banding, highlight_anti_banding
                    )
                
                processed_frames.append(processed_frame)
        
        return torch.stack(processed_frames, dim=0)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Colour Match First Frames": VTS_ColourMatchFirstFrames
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Colour Match First Frames": "VTS Colour Match First Frames"
}
