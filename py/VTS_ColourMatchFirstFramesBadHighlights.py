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


class VTS_ColourMatchFirstFramesBadHighlights:
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
                "shadow_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Strength of shadow correction - automatically brightens dark shadows and restores saturation. 0.0=no correction, 1.0=full correction, >1.0=over-correction"}),
                "highlight_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Strength of highlight correction - automatically tones down blown highlights and restores detail. 0.0=no correction, 1.0=full correction, >1.0=over-correction"}),
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
- Prevents highlights from becoming blown out
- Uses the same proven first-frames approach for tonal stability
- Individual strength controls for shadow_strength and highlight_strength
- Advanced anti-banding with smooth zone transitions and local adaptive corrections
- Edge-preserving smoothing to eliminate banding artifacts while preserving detail

Anti-Banding Features:
- Smooth transition masks instead of hard thresholds
- Local neighborhood analysis for spatially-varying corrections
- Separate anti-banding controls for shadows and highlights
- Edge-preserving smoothing that respects image structure
"""

    CATEGORY = "VTS"

    def colormatch(self, image_target, method, passthrough, strength=1.0, numberOfFirstFrames=20, 
                   contrast_stabilization=False, shadow_threshold=0.3, highlight_threshold=0.7, shadow_strength=0.8, highlight_strength=0.8,
                   shadow_anti_banding=0.3, highlight_anti_banding=0.2, editInPlace=False, gc_interval=50):
        if passthrough:
            print("VTS_ColourMatchFirstFrames - passthrough is True, returning original image_target without processing")
            return (image_target,)
        
        mode_desc = "color matching"
        if contrast_stabilization:
            mode_desc += " + contrast stabilization"
        
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

    def calculate_brightness_signature_with_zones(self, frame, shadow_threshold=0.3, highlight_threshold=0.7):
        """Calculate brightness signature including shadow/highlight zone analysis"""
        lum = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        
        # Define tonal zones
        shadow_mask = lum < shadow_threshold
        midtone_mask = (lum >= shadow_threshold) & (lum <= highlight_threshold)
        highlight_mask = lum > highlight_threshold
        
        # Standard brightness signature
        hist = torch.histc(lum.flatten(), bins=16, min=0, max=1)
        hist_norm = hist / (hist.sum() + 1e-8)
        mean_lum = lum.mean().item()
        std_lum = lum.std().item()
        percentiles = torch.quantile(lum.flatten(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        
        # Zone-specific analysis
        shadow_stats = self.analyze_zone(frame, shadow_mask, "shadows")
        highlight_stats = self.analyze_zone(frame, highlight_mask, "highlights")
        
        # Contrast characteristics
        contrast_ratio = percentiles[4] - percentiles[0]  # 90th - 10th percentile
        
        return {
            'histogram': hist_norm,
            'mean': mean_lum,
            'std': std_lum,
            'percentiles': percentiles,
            'shadow_stats': shadow_stats,
            'highlight_stats': highlight_stats,
            'contrast_ratio': contrast_ratio.item()
        }

    def analyze_zone(self, frame, mask, zone_name):
        """Analyze specific tonal zone characteristics"""
        if mask.sum() == 0:
            return {
                'area_ratio': 0.0,
                'mean_luminance': 0.5,
                'mean_color': torch.tensor([0.5, 0.5, 0.5]),
                'saturation': 0.0
            }
        
        zone_pixels = frame[mask]
        zone_lum = (0.299 * zone_pixels[..., 0] + 0.587 * zone_pixels[..., 1] + 0.114 * zone_pixels[..., 2])
        
        # Calculate saturation (approximate)
        zone_max, _ = zone_pixels.max(dim=-1)
        zone_min, _ = zone_pixels.min(dim=-1)
        saturation = ((zone_max - zone_min) / (zone_max + 1e-8)).mean()
        
        return {
            'area_ratio': (mask.sum().float() / mask.numel()).item(),
            'mean_luminance': zone_lum.mean().item(),
            'mean_color': zone_pixels.mean(dim=0),
            'saturation': saturation.item()
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

    def apply_contrast_stabilization(self, reference_frame, current_frame, shadow_strength=0.8, highlight_strength=0.8, shadow_threshold=0.3, highlight_threshold=0.7, shadow_anti_banding=0.3, highlight_anti_banding=0.2):
        """Apply smooth contrast stabilization to prevent shadow/highlight drift with anti-banding measures"""
        curr_lum = 0.299 * current_frame[..., 0] + 0.587 * current_frame[..., 1] + 0.114 * current_frame[..., 2]
        ref_lum = 0.299 * reference_frame[..., 0] + 0.587 * reference_frame[..., 1] + 0.114 * reference_frame[..., 2]
        
        result = current_frame.clone()
        corrections_applied = []
        
        # Create smooth transition masks instead of hard thresholds
        # Shadow mask: strong at 0, fades to 0 at shadow_threshold, extends slightly beyond
        shadow_fade_end = shadow_threshold + 0.1
        shadow_mask = torch.clamp((shadow_fade_end - curr_lum) / (shadow_fade_end - 0.0), 0.0, 1.0)
        shadow_mask = shadow_mask * (curr_lum < shadow_fade_end).float()
        
        # Highlight mask: starts at highlight_threshold, strong at 1.0, with smooth transition
        highlight_fade_start = highlight_threshold - 0.1
        highlight_mask = torch.clamp((curr_lum - highlight_fade_start) / (1.0 - highlight_fade_start), 0.0, 1.0)
        highlight_mask = highlight_mask * (curr_lum > highlight_fade_start).float()
        
        # Shadow stabilization with local adaptive correction
        if shadow_mask.sum() > 0:
            # Use local neighborhood analysis instead of global means
            shadow_correction = self.calculate_local_correction(
                current_frame, reference_frame, curr_lum, ref_lum, shadow_mask, 
                is_shadow=True, strength=shadow_strength
            )
            
            # Apply correction with smooth blending
            for c in range(3):  # RGB channels
                correction_amount = shadow_correction[..., c] * shadow_mask
                result[..., c] = result[..., c] + correction_amount
            
            corrections_applied.append("shadows")
        
        # Highlight stabilization with local adaptive correction
        if highlight_mask.sum() > 0:
            # Use local neighborhood analysis for highlights
            highlight_correction = self.calculate_local_correction(
                current_frame, reference_frame, curr_lum, ref_lum, highlight_mask,
                is_shadow=False, strength=highlight_strength
            )
            
            # Apply correction with smooth blending
            for c in range(3):  # RGB channels
                correction_amount = highlight_correction[..., c] * highlight_mask
                result[..., c] = result[..., c] + correction_amount
            
            corrections_applied.append("highlights")
        
        # Apply separate anti-banding smoothing for shadows and highlights
        if shadow_anti_banding > 0 and shadow_mask.sum() > 0:
            result = self.apply_zone_smoothing(result, current_frame, shadow_mask, shadow_anti_banding, "shadow")
        
        if highlight_anti_banding > 0 and highlight_mask.sum() > 0:
            result = self.apply_zone_smoothing(result, current_frame, highlight_mask, highlight_anti_banding, "highlight")
        
        result = torch.clamp(result, 0.0, 1.0)
        return result, corrections_applied

    def calculate_local_correction(self, current_frame, reference_frame, curr_lum, ref_lum, mask, is_shadow=True, strength=1.0):
        """Fast vectorized correction to avoid banding - replaces slow pixel-by-pixel processing"""
        h, w = curr_lum.shape
        
        # Early exit if no significant pixels
        significant_pixels = mask > 0.1
        if significant_pixels.sum() == 0:
            return torch.zeros_like(current_frame)
        
        # Use smaller patches for efficiency while maintaining quality
        patch_size = 7 if is_shadow else 11  # Much smaller than original 15/25
        padding = patch_size // 2
        
        # Vectorized approach: process in patches instead of per-pixel
        correction = torch.zeros_like(current_frame)
        
        # Create padded versions for neighborhood access
        curr_padded = torch.nn.functional.pad(curr_lum.unsqueeze(0).unsqueeze(0), 
                                              (padding, padding, padding, padding), mode='reflect')
        ref_padded = torch.nn.functional.pad(ref_lum.unsqueeze(0).unsqueeze(0), 
                                             (padding, padding, padding, padding), mode='reflect')
        
        # Get all neighborhoods at once using unfold
        curr_neighborhoods = torch.nn.functional.unfold(curr_padded, patch_size, stride=1)
        ref_neighborhoods = torch.nn.functional.unfold(ref_padded, patch_size, stride=1)
        
        # Reshape to [patch_pixels, h, w]
        patch_pixels = patch_size * patch_size
        curr_neighborhoods = curr_neighborhoods.view(patch_pixels, h, w)
        ref_neighborhoods = ref_neighborhoods.view(patch_pixels, h, w)
        
        # Vectorized local means calculation
        curr_local_means = curr_neighborhoods.mean(dim=0)  # [h, w]
        ref_local_means = ref_neighborhoods.mean(dim=0)    # [h, w]
        
        # Avoid division by zero
        valid_mask = (curr_local_means > 1e-6) & significant_pixels
        if not valid_mask.any():
            return correction
        
        # Calculate luminance correction factors for all pixels at once
        lum_factors = ref_local_means / (curr_local_means + 1e-8)
        
        # Apply zone-specific clamping
        if is_shadow:
            lum_factors = torch.clamp(lum_factors, 0.85, 1.4)
            mix_ratio = 0.4
        else:
            lum_factors = torch.clamp(lum_factors, 0.6, 1.15)
            mix_ratio = 0.6
        
        # Simple global color correction (much faster than per-pixel patches)
        curr_mean_color = current_frame[significant_pixels].mean(dim=0)
        ref_mean_color = reference_frame[significant_pixels].mean(dim=0)
        global_color_correction = ref_mean_color - curr_mean_color
        
        # Apply corrections only where valid
        for c in range(3):
            # Luminance-based correction
            lum_correction = current_frame[..., c] * (lum_factors - 1.0)
            
            # Combine with global color correction
            total_correction = mix_ratio * lum_correction + (1.0 - mix_ratio) * global_color_correction[c]
            
            # Apply only where mask is significant
            correction[..., c] = strength * total_correction * valid_mask.float()
        
        return correction

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
                                     contrast_stabilization=False, shadow_threshold=0.3, highlight_threshold=0.7, shadow_strength=0.8, highlight_strength=0.8,
                                     shadow_anti_banding=0.3, highlight_anti_banding=0.2):
        """
        Enhanced processing with optional contrast stabilization using separate shadow and highlight strength controls with anti-banding
        """
        if color_match_strength <= 0.0 and not contrast_stabilization:
            return decoded_frames
        
        processed_frames = []
        
        for i, current_frame in enumerate(decoded_frames):
            current_frame_batch = current_frame.unsqueeze(0)  # Add batch dimension
            
            if self.brightness_lookup is None or len(self.brightness_lookup) < self.numberOfFirstFrames:
                # Build enhanced brightness lookup with tonal zone analysis
                if contrast_stabilization:
                    signature = self.calculate_brightness_signature_with_zones(current_frame, shadow_threshold, highlight_threshold)
                    zone_info = f", shadows: {signature['shadow_stats']['area_ratio']:.2f}, highlights: {signature['highlight_stats']['area_ratio']:.2f}"
                else:
                    signature = self.calculate_brightness_signature(current_frame)
                    zone_info = ""
                
                self.brightness_lookup.append({
                    'frame': current_frame.clone(),
                    'signature': signature,
                    'frame_index': i
                })
                print(f"Frame {i}: Added to lookup (lum: {signature['mean']:.3f}, std: {signature['std']:.3f}{zone_info})")
                processed_frames.append(current_frame)
            else:
                # Find best matching reference frame
                if contrast_stabilization:
                    target_signature = self.calculate_brightness_signature_with_zones(current_frame, shadow_threshold, highlight_threshold)
                else:
                    target_signature = self.calculate_brightness_signature(current_frame)
                
                best_match = self.find_best_brightness_match(target_signature, self.brightness_lookup)
                
                if best_match is not None:
                    reference_frame = best_match['frame'].unsqueeze(0)
                    processed_frame = current_frame.clone()
                    applied_corrections = []
                    
                    # Apply color matching if enabled
                    if color_match_strength > 0.0:
                        color_match_result = colormatch(reference_frame, current_frame_batch, color_match_method, color_match_strength, editInPlace, gc_interval)
                        processed_frame = color_match_result[0][0]
                        applied_corrections.append("color_match")
                    
                    # Apply contrast stabilization if enabled
                    if contrast_stabilization:
                        processed_frame, corrections = self.apply_contrast_stabilization(
                            best_match['frame'], processed_frame, shadow_strength, highlight_strength, 
                            shadow_threshold, highlight_threshold, shadow_anti_banding, highlight_anti_banding
                        )
                        if corrections:
                            applied_corrections.extend(corrections)
                    
                    correction_desc = " + ".join(applied_corrections) if applied_corrections else "none"
                    print(f"Frame {i}: Matched with ref {best_match['frame_index']} (lum: {target_signature['mean']:.3f}) - applied: {correction_desc}")
                    
                    processed_frames.append(processed_frame)
                else:
                    print(f"Frame {i}: No suitable match found, using original frame")
                    processed_frames.append(current_frame)
        
        return torch.stack(processed_frames, dim=0)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Colour Match First Frames Bad Highlights": VTS_ColourMatchFirstFramesBadHighlights
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Colour Match First Frames Bad Highlights": "VTS Colour Match First Frames Bad Highlights"
}
