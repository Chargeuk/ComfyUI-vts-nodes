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


class VTS_ColourMatchAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
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
                "editInPlace": ("BOOLEAN", {"default": False, "tooltip": "When true, modify the input image_target tensor directly instead of creating a new tensor"}),
                "gc_interval": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1, "tooltip": "Garbage collection interval. Set to 0 to disable automatic garbage collection. For large batches, lower values can help manage memory"}),
                "color_match_scene_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Threshold for scene change detection"}),
                "color_match_adaptive": ("BOOLEAN", {"default": True, "tooltip": "Use adaptive color matching strength"}),
                "temporal_smoothing": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Temporal smoothing factor"}),
                "color_reference_method": (["first_frame", "rolling_average", "previous_frame", "firstFramesHistogram"], {"default": "rolling_average"}),
                "numberOfFirstFrames": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1, "tooltip": "Number of first frames to use as brightness reference library (only used with firstFramesHistogram method)"}),
                "color_correction_method": (["frame_by_frame", "advanced"], {"default": "frame_by_frame", "tooltip": "Choose color correction method: frame_by_frame (original) or advanced (luminance-preserving)"}),
                "requiredTotalCorrection": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Minimum total correction needed to apply advanced color correction"}),
                "whiteBalanceMultiply": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Multiplier for white balance correction strength"}),
                "luminanceMultiply": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Multiplier for luminance-preserving correction strength"}),
                "mixedLightingMultiply": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Multiplier for zone-based correction strength"}),
                "whiteBalancePercentage": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 1000.0, "step": 1.0, "tooltip": "Percentage of white balance correction to apply"}),
                "luminancePercentage": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 1000.0, "step": 1.0, "tooltip": "Percentage of luminance-preserving correction to apply"}),
                "mixedLightingPercentage": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 1000.0, "step": 1.0, "tooltip": "Percentage of zone-based correction to apply"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    DESCRIPTION = """
color-matcher enables color transfer across images which comes in handy for automatic  
color-grading of photographs, paintings and film sequences as well as light-field  
and stopmotion corrections.  

The methods behind the mappings are based on the approach from Reinhard et al.,  
the Monge-Kantorovich Linearization (MKL) as proposed by Pitie et al. and our analytical solution  
to a Multi-Variate Gaussian Distribution (MVGD) transfer in conjunction with classical histogram   
matching. As shown below our HM-MVGD-HM compound outperforms existing methods.   
https://github.com/hahnec/color-matcher/

"""

    CATEGORY = "VTS"

    def colormatch(self, image_ref, image_target, method, passthrough, strength=1.0, editInPlace=False, gc_interval=50,
                   color_match_scene_threshold=0.3, color_match_adaptive=True, temporal_smoothing=0.1, color_reference_method="rolling_average", numberOfFirstFrames=20, color_correction_method="frame_by_frame",
                    requiredTotalCorrection=0.1, whiteBalanceMultiply=0.8, luminanceMultiply=0.7, mixedLightingMultiply=0.8,
                    whiteBalancePercentage=100.0, luminancePercentage=100.0, mixedLightingPercentage=100.0):
        if passthrough:
            print("VTS_ColourMatchADV - passthrough is True, returning original image_target without processing")
            return (image_target,)
        print(f"VTS_ColourMatchADV - passthrough is False, processing images. method: {method}, strength: {strength}, editInPlace: {editInPlace}, gc_interval: {gc_interval}")
        # output = colormatch(image_ref, image_target, method, strength, editInPlace, gc_interval)

        self.color_reference_method = color_reference_method
        self.color_reference_buffer = []
        self.original_reference_buffer = []  # Buffer for original frames (scene change detection)
        self.buffer_size = 5  # Keep last 5 frames as references
        self.brightness_lookup = []  # For firstFramesHistogram method
        self.numberOfFirstFrames = numberOfFirstFrames
        
        # Apply color matching to prefix samples based on selected method
        if color_correction_method == "advanced":
            print(f"VTS_ColourMatchADV - Applying advanced color correction to prefix samples")
            output = self.process_batch_with_advanced_color_correction(
                image_target,
                strength,
                method,
                color_match_adaptive,
                color_match_scene_threshold,
                temporal_smoothing,
                image_ref,
                requiredTotalCorrection,
                whiteBalanceMultiply,
                luminanceMultiply,
                mixedLightingMultiply,
                whiteBalancePercentage,
                luminancePercentage,
                mixedLightingPercentage
            )
        else:
            print(f"VTS_ColourMatchADV - Applying frame-by-frame color matching to prefix samples")
            output = self.process_batch_with_frame_by_frame_colormatch(
                image_target,
                strength,
                method,
                color_match_adaptive,
                color_match_scene_threshold,
                temporal_smoothing,
                image_ref
            )
        print(f"VTS_ColourMatchADV - finished processing images. method: {method}, strength: {strength}, editInPlace: {editInPlace}, gc_interval: {gc_interval}")
        return (output,)
    
    def detect_scene_change(self, prev_frame, current_frame, threshold=0.3):
        sceneChangeRgb = self.detect_scene_change_rgb(prev_frame, current_frame, threshold)
        sceneChangeLab = self.detect_scene_change_lab(prev_frame, current_frame, threshold)
        return max(sceneChangeRgb, sceneChangeLab) * 20

    def detect_scene_change_rgb(self, prev_frame, current_frame, threshold=0.3):
        """
        Detect if there's a significant scene change between frames
        Returns a value between 0 (no change) and 1 (complete change)
        """
        # Use RGB directly for simpler, more predictable results
        prev_flat = prev_frame.flatten()
        curr_flat = current_frame.flatten()
        
        # Calculate histogram differences
        prev_hist = torch.histc(prev_flat, bins=64, min=0, max=1)
        curr_hist = torch.histc(curr_flat, bins=64, min=0, max=1)
        
        # Normalize histograms
        prev_hist_norm = prev_hist / (prev_hist.sum() + 1e-8)
        curr_hist_norm = curr_hist / (curr_hist.sum() + 1e-8)
        
        hist_diff = torch.abs(prev_hist_norm - curr_hist_norm).sum() * 0.5
        
        # Calculate mean difference per channel
        prev_mean = prev_frame.mean(dim=[0, 1])
        curr_mean = current_frame.mean(dim=[0, 1])
        mean_diff = torch.abs(prev_mean - curr_mean).mean()
        
        # Combine metrics
        scene_change_score = 0.7 * hist_diff + 0.3 * mean_diff
        calculatedDiff = scene_change_score.item()
        print(f"Scene change RGB score: {calculatedDiff:.3f} (hist: {hist_diff:.3f}, mean: {mean_diff:.3f}), threshold: {threshold:.3f}")
        
        return min(calculatedDiff, 1.0)


    def detect_scene_change_lab(self, prev_frame, current_frame, threshold=0.3):
        """
        Detect if there's a significant scene change between frames
        Returns a value between 0 (no change) and 1 (complete change)
        """
        # Convert to LAB color space for perceptual difference
        prev_lab = self.rgb_to_lab(prev_frame)
        curr_lab = self.rgb_to_lab(current_frame)
        
        # Calculate histogram differences with fewer bins for better sensitivity
        prev_flat = prev_lab.flatten()
        curr_flat = curr_lab.flatten()
        
        # Use 64 bins instead of 256 for better population per bin
        prev_hist = torch.histc(prev_flat, bins=64, min=0, max=1)
        curr_hist = torch.histc(curr_flat, bins=64, min=0, max=1)
        
        # Normalize histograms to probabilities (sum = 1)
        prev_hist_norm = prev_hist / (prev_hist.sum() + 1e-8)  # Add epsilon for safety
        curr_hist_norm = curr_hist / (curr_hist.sum() + 1e-8)
        
        # Use sum instead of mean for histogram difference (gives better range)
        hist_diff = torch.abs(prev_hist_norm - curr_hist_norm).sum() * 0.5
        
        # Calculate mean difference per channel for better sensitivity
        prev_mean = prev_lab.mean(dim=[0, 1])  # [L, A, B] means
        curr_mean = curr_lab.mean(dim=[0, 1])  # [L, A, B] means
        mean_diff = torch.abs(prev_mean - curr_mean).mean()
        
        # Weight histogram difference more heavily for scene changes
        scene_change_score = 0.7 * hist_diff + 0.3 * mean_diff
        calculatedDiff = scene_change_score.item()
        print(f"Scene change LAB score: {calculatedDiff:.3f} (hist: {hist_diff:.3f}, mean: {mean_diff:.3f}), threshold: {threshold:.3f}")
        
        return min(calculatedDiff, 1.0)
        

    def rgb_to_lab(self, rgb_tensor):
        """Simple RGB to LAB conversion approximation"""
        # This is a simplified version - you might want to use a proper color space conversion
        r, g, b = rgb_tensor[..., 0], rgb_tensor[..., 1], rgb_tensor[..., 2]
        l = 0.299 * r + 0.587 * g + 0.114 * b  # Range: 0-1
        a = (r - g) * 0.5 + 0.5                # Range: 0-1 (shifted)
        b_comp = (r + g - 2 * b) * 0.25 + 0.5  # Range: 0-1 (shifted)
        return torch.stack([l, a, b_comp], dim=-1)

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

    def temporal_color_smooth(self, current_frames, prev_frames, smoothing_factor=0.1):
        """
        Apply temporal smoothing to reduce sudden color changes
        Uses direct RGB smoothing to avoid color space conversion issues
        """
        if prev_frames is None or smoothing_factor <= 0.0:
            return current_frames
        
        # Get the right number of previous frames to match current frames
        prev_frames_matched = prev_frames[-current_frames.shape[0]:]
        
        # Direct RGB smoothing (safer, avoids color space conversion issues)
        smoothed_rgb = (1 - smoothing_factor) * current_frames + smoothing_factor * prev_frames_matched
        
        # Clamp values to valid range
        smoothed_rgb = torch.clamp(smoothed_rgb, 0.0, 1.0)
        
        return smoothed_rgb

    def process_batch_with_frame_by_frame_colormatch(self, decoded_frames, color_match_strength, color_match_method, 
                                                    color_match_adaptive, color_match_scene_threshold, temporal_smoothing, color_match_source=None):
        """
        Process each frame in a batch individually for more precise color matching
        Updates references with the corrected frames
        """
        if color_match_strength <= 0.0:
            return decoded_frames
        
        processed_frames = []
        original_frames = list(decoded_frames)  # Keep track of original frames for scene change detection
        
        for i, current_frame in enumerate(decoded_frames):
            current_frame_batch = current_frame.unsqueeze(0)  # Add batch dimension
            
            # Handle firstFramesHistogram method - build lookup in first frames, use it for remaining frames
            if self.color_reference_method == "firstFramesHistogram":
                if i < self.numberOfFirstFrames:
                    # Build brightness lookup from first frames (no color matching applied)
                    signature = self.calculate_brightness_signature(current_frame)
                    self.brightness_lookup.append({
                        'frame': current_frame.clone(),
                        'signature': signature,
                        'frame_index': i
                    })
                    print(f"Frame {i}: Added to brightness lookup (mean luminance: {signature['mean']:.3f})")
                    processed_frames.append(current_frame)
                    continue
                else:
                    # Use brightness lookup to find best matching reference frame
                    target_signature = self.calculate_brightness_signature(current_frame)
                    best_match = self.find_best_brightness_match(target_signature, self.brightness_lookup)
                    
                    if best_match is not None:
                        reference_frame = best_match['frame'].unsqueeze(0)
                        print(f"Frame {i}: Matched with reference frame {best_match['frame_index']} (distance calculated from histogram)")
                    else:
                        print(f"Frame {i}: No suitable match found in brightness lookup, using current frame")
                        reference_frame = current_frame_batch.clone()
            else:
                # Original reference frame determination logic
                # Determine reference frame for this specific frame (use CORRECTED frames)
                reference_frame = None
                
                if self.color_reference_method == "rolling_average" and len(self.color_reference_buffer) > 0:
                    # Use rolling average as reference
                    reference_frame = torch.stack(self.color_reference_buffer).mean(dim=0).unsqueeze(0)
                elif self.color_reference_method == "previous_frame":
                    if i > 0:
                        # Use previous CORRECTED frame in current batch
                        reference_frame = processed_frames[-1].unsqueeze(0)
                    elif len(self.color_reference_buffer) > 0:
                        # Use last CORRECTED frame from buffer if this is first frame in batch
                        reference_frame = self.color_reference_buffer[-1].unsqueeze(0)
                else:
                    # Default to first frame method or provided color_match_source
                    if color_match_source is not None:
                        reference_frame = color_match_source
                    elif len(self.color_reference_buffer) > 0:
                        reference_frame = self.color_reference_buffer[0].unsqueeze(0)
                    elif i > 0:
                        reference_frame = processed_frames[0].unsqueeze(0)
                
                # If no reference available, use current frame as reference (no change)
                if reference_frame is None:
                    print(f"Frame {i}: No reference frame available, using current frame as reference, so no change")
                    reference_frame = current_frame_batch.clone()
            
            adaptive_strength = color_match_strength
                
            # Set initial color_match_source if not set yet and this is the first frame
            if i == 0 and color_match_source is None:
                color_match_source = current_frame_batch.clone()
            
            # Apply adaptive color matching if enabled
            if color_match_adaptive:
                scene_change_score = 0.0
                if i > 0:
                    # Compare ORIGINAL frames for scene change detection
                    scene_change_score = self.detect_scene_change(
                        original_frames[i-1].unsqueeze(0),  # Previous ORIGINAL frame
                        current_frame_batch,                # Current ORIGINAL frame
                        color_match_scene_threshold
                    )
                elif hasattr(self, 'original_reference_buffer') and len(self.original_reference_buffer) > 0:
                    # Compare with last ORIGINAL frame from buffer
                    scene_change_score = self.detect_scene_change(
                        self.original_reference_buffer[-1].unsqueeze(0),  # Last ORIGINAL frame from buffer
                        current_frame_batch,                              # Current ORIGINAL frame
                        color_match_scene_threshold
                    )
                
                # Use threshold to determine if scene change is significant enough to reduce color matching
                if scene_change_score > color_match_scene_threshold:
                    # Calculate how much the scene change exceeds the threshold (0.0 to 1.0 range)
                    # Handle edge case where threshold is 1.0 (divisor would be 0)
                    if color_match_scene_threshold >= 1.0:
                        excess_change = 1.0  # Maximum reduction when threshold is at maximum
                    else:
                        excess_change = min((scene_change_score - color_match_scene_threshold) / (1.0 - color_match_scene_threshold), 1.0)
                    
                    # Gradually reduce strength from full strength to minimum (e.g., 20% of original)
                    min_strength_ratio = 0.2  # Don't go below 20% of original strength
                    reduction_factor = 1.0 - (excess_change * (1.0 - min_strength_ratio))
                    adaptive_strength = color_match_strength * reduction_factor
                else:
                    # Keep original strength for minor changes
                    adaptive_strength = color_match_strength
                
                print(f"Frame {i}: Scene change score: {scene_change_score:.3f}, threshold: {color_match_scene_threshold:.3f}, adaptive strength: {adaptive_strength:.3f}")
            
            # Apply color matching using CORRECTED reference frame
            if adaptive_strength > 0.0:
                print(f"Frame {i}: Applying color matching with strength {adaptive_strength:.3f} using method {color_match_method}")
                color_match_result = colormatch(reference_frame, current_frame_batch, color_match_method, adaptive_strength)
                processed_frame = color_match_result[0][0]  # Remove batch dimension
            else:
                print(f"Frame {i}: No color matching applied, adaptive strength is 0.0!!!!")
                processed_frame = current_frame
            
            # Apply temporal smoothing if enabled
            if temporal_smoothing > 0.0 and i > 0:
                processed_frame_batch = processed_frame.unsqueeze(0)
                prev_frame_batch = processed_frames[-1].unsqueeze(0)
                smoothed_frame = self.temporal_color_smooth(processed_frame_batch, prev_frame_batch, temporal_smoothing)
                processed_frame = smoothed_frame[0]
            
            processed_frames.append(processed_frame)
            
            # Update rolling buffer with CORRECTED frame for color matching (skip for firstFramesHistogram)
            if self.color_reference_method == "rolling_average":
                if len(self.color_reference_buffer) >= self.buffer_size:
                    self.color_reference_buffer.pop(0)
                self.color_reference_buffer.append(processed_frame.clone())
            
            # Also maintain original frame buffer for scene change detection (skip for firstFramesHistogram)
            if self.color_reference_method != "firstFramesHistogram":
                if not hasattr(self, 'original_reference_buffer'):
                    self.original_reference_buffer = []
                if len(self.original_reference_buffer) >= self.buffer_size:
                    self.original_reference_buffer.pop(0)
                self.original_reference_buffer.append(current_frame.clone())
        
        return torch.stack(processed_frames, dim=0)

    def apply_white_balance_correction(self, reference_frame, current_frame, strength=0.5):
        """
        Apply white balance correction instead of global color matching
        Preserves scene brightness characteristics while correcting color cast
        """
        # Calculate average color cast difference
        ref_mean = reference_frame.mean(dim=[0, 1])  # [R, G, B] averages
        curr_mean = current_frame.mean(dim=[0, 1])   # [R, G, B] averages
        
        # Calculate white balance correction (ratios, not absolute differences)
        wb_ratios = ref_mean / (curr_mean + 1e-8)
        
        # Limit correction to prevent over-correction
        wb_ratios = torch.clamp(wb_ratios, 0.8, 1.2)
        
        # Apply white balance with strength control
        corrected = current_frame * (1.0 + strength * (wb_ratios - 1.0))
        
        # Clamp to valid range
        return torch.clamp(corrected, 0.0, 1.0)

    def apply_luminance_preserving_correction(self, reference_frame, current_frame, strength=0.5):
        """
        Correct color while preserving original luminance structure
        Prevents washing out of dark subjects
        """
        # Convert to YUV-like space (luminance + chrominance)
        def rgb_to_yuv(rgb):
            y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            u = rgb[..., 2] - y  # Blue chrominance
            v = rgb[..., 0] - y  # Red chrominance
            return y, u, v
        
        def yuv_to_rgb(y, u, v):
            r = y + v
            g = y - 0.299 * v / 0.587 - 0.114 * u / 0.587
            b = y + u
            return torch.stack([r, g, b], dim=-1)
        
        # Extract luminance and chrominance
        curr_y, curr_u, curr_v = rgb_to_yuv(current_frame)
        ref_y, ref_u, ref_v = rgb_to_yuv(reference_frame)
        
        # Calculate average chrominance shift (preserve local luminance)
        ref_u_mean = ref_u.mean()
        ref_v_mean = ref_v.mean()
        curr_u_mean = curr_u.mean()
        curr_v_mean = curr_v.mean()
        
        # Apply chrominance correction while keeping original luminance
        corrected_u = curr_u + strength * (ref_u_mean - curr_u_mean)
        corrected_v = curr_v + strength * (ref_v_mean - curr_v_mean)
        
        # Reconstruct RGB with original luminance
        corrected_rgb = yuv_to_rgb(curr_y, corrected_u, corrected_v)
        
        return torch.clamp(corrected_rgb, 0.0, 1.0)

    def apply_zone_based_histogram_matching(self, reference_frame, current_frame, strength=0.5):
        """
        Apply histogram matching separately for different luminance zones
        Prevents dark subjects from being washed out
        """
        # Calculate luminance
        ref_lum = 0.299 * reference_frame[..., 0] + 0.587 * reference_frame[..., 1] + 0.114 * reference_frame[..., 2]
        curr_lum = 0.299 * current_frame[..., 0] + 0.587 * current_frame[..., 1] + 0.114 * current_frame[..., 2]
        
        # Define luminance zones
        dark_mask = curr_lum < 0.3
        mid_mask = (curr_lum >= 0.3) & (curr_lum < 0.7)
        bright_mask = curr_lum >= 0.7
        
        result = current_frame.clone()
        
        # Apply different corrections to different zones
        for mask, zone_name in [(dark_mask, "dark"), (mid_mask, "mid"), (bright_mask, "bright")]:
            if mask.sum() > 0:  # If zone has pixels
                # Extract pixels in this zone
                zone_curr = current_frame[mask]
                
                # Find corresponding reference zone (with some tolerance)
                if zone_name == "dark":
                    ref_zone_mask = ref_lum < 0.4  # Slightly larger range for reference
                elif zone_name == "mid":
                    ref_zone_mask = (ref_lum >= 0.25) & (ref_lum < 0.75)
                else:  # bright
                    ref_zone_mask = ref_lum >= 0.6
                
                if ref_zone_mask.sum() > 0:
                    ref_zone = reference_frame[ref_zone_mask]
                    
                    # Calculate color correction for this zone only
                    zone_curr_mean = zone_curr.mean(dim=0)
                    ref_zone_mean = ref_zone.mean(dim=0)
                    
                    # Apply gentle correction
                    correction = strength * (ref_zone_mean - zone_curr_mean)
                    result[mask] = torch.clamp(zone_curr + correction, 0.0, 1.0)
        
        return result

    def analyze_luminance_characteristics(self, reference_frame, current_frame):
        """Analyze frame characteristics to choose appropriate correction method"""
        ref_lum = 0.299 * reference_frame[..., 0] + 0.587 * reference_frame[..., 1] + 0.114 * reference_frame[..., 2]
        curr_lum = 0.299 * current_frame[..., 0] + 0.587 * current_frame[..., 1] + 0.114 * current_frame[..., 2]
        
        # Calculate luminance histogram changes
        ref_hist = torch.histc(ref_lum.flatten(), bins=32, min=0, max=1)
        curr_hist = torch.histc(curr_lum.flatten(), bins=32, min=0, max=1)
        
        ref_hist_norm = ref_hist / (ref_hist.sum() + 1e-8)
        curr_hist_norm = curr_hist / (curr_hist.sum() + 1e-8)
        
        # Check for changes in dark and bright regions
        dark_region_change = torch.abs(ref_hist_norm[:10] - curr_hist_norm[:10]).sum()
        bright_region_change = torch.abs(ref_hist_norm[22:] - curr_hist_norm[22:]).sum()
        contrast = curr_lum.std()

        color_cast_check = torch.abs(current_frame.mean(dim=[0,1]) - reference_frame.mean(dim=[0,1])).max()

        luminance_variance = torch.abs(ref_lum.std() - curr_lum.std())
        print(f"Dark region change: {dark_region_change:.3f}, Bright region change: {bright_region_change:.3f}, Luminance Variance: {luminance_variance:.3f}, contrast: {contrast:.3f}, color_cast_check: {color_cast_check:.3f}")

        return {
            # Boolean flags (for fallback logic)
            'has_high_contrast': (contrast > 0.18),
            'luminance_variance_high': (luminance_variance > 0.015),
            'color_cast_detected': (color_cast_check > 0.01),
            'mixed_lighting': (dark_region_change > 0.03 and bright_region_change > 0.03),
            
            # Raw values (for weighted blending)
            'dark_region_change': dark_region_change.item(),
            'bright_region_change': bright_region_change.item(),
            'contrast': contrast.item(),
            'color_cast_check': color_cast_check.item(),
            'luminance_variance': luminance_variance.item()
        }

    def process_batch_with_advanced_color_correction(self, decoded_frames, color_match_strength, color_match_method, 
                                                   color_match_adaptive, color_match_scene_threshold, temporal_smoothing,
                                                   color_match_source=None, requiredTotalCorrection=0.1,
                                                   whiteBalanceMultiply=0.8, luminanceMultiply=0.7, mixedLightingMultiply=0.8,
                                                   whiteBalancePercentage=100.0, luminancePercentage=100.0, mixedLightingPercentage=100.0):
        """
        Enhanced processing with multiple color correction methods to prevent washing out
        """
        if color_match_strength <= 0.0:
            return decoded_frames
        
        processed_frames = []
        original_frames = list(decoded_frames)  # Keep track of original frames for scene change detection
        
        for i, current_frame in enumerate(decoded_frames):
            current_frame_batch = current_frame.unsqueeze(0)  # Add batch dimension
            
            # Handle firstFramesHistogram method - build lookup in first frames, use it for remaining frames
            if self.color_reference_method == "firstFramesHistogram":
                if i < self.numberOfFirstFrames:
                    # Build brightness lookup from first frames (no color matching applied)
                    signature = self.calculate_brightness_signature(current_frame)
                    self.brightness_lookup.append({
                        'frame': current_frame.clone(),
                        'signature': signature,
                        'frame_index': i
                    })
                    print(f"Frame {i}: Added to brightness lookup (mean luminance: {signature['mean']:.3f})")
                    processed_frames.append(current_frame)
                    continue
                else:
                    # Use brightness lookup to find best matching reference frame
                    target_signature = self.calculate_brightness_signature(current_frame)
                    best_match = self.find_best_brightness_match(target_signature, self.brightness_lookup)
                    
                    if best_match is not None:
                        reference_frame = best_match['frame'].unsqueeze(0)
                        print(f"Frame {i}: Matched with reference frame {best_match['frame_index']} (distance calculated from histogram)")
                    else:
                        print(f"Frame {i}: No suitable match found in brightness lookup, using current frame")
                        reference_frame = current_frame_batch.clone()
            else:
                # Original reference frame determination logic
                # Determine reference frame for this specific frame (use CORRECTED frames)
                reference_frame = None
                
                if self.color_reference_method == "rolling_average" and len(self.color_reference_buffer) > 0:
                    # Use rolling average as reference
                    reference_frame = torch.stack(self.color_reference_buffer).mean(dim=0).unsqueeze(0)
                elif self.color_reference_method == "previous_frame":
                    if i > 0:
                        # Use previous CORRECTED frame in current batch
                        reference_frame = processed_frames[-1].unsqueeze(0)
                    elif len(self.color_reference_buffer) > 0:
                        # Use last CORRECTED frame from buffer if this is first frame in batch
                        reference_frame = self.color_reference_buffer[-1].unsqueeze(0)
                else:
                    # Default to first frame method or provided color_match_source
                    if color_match_source is not None:
                        reference_frame = color_match_source
                    elif len(self.color_reference_buffer) > 0:
                        reference_frame = self.color_reference_buffer[0].unsqueeze(0)
                    elif i > 0:
                        reference_frame = processed_frames[0].unsqueeze(0)
                
                # If no reference available, use current frame as reference (no change)
                if reference_frame is None:
                    print(f"Frame {i}: No reference frame available, using current frame as reference, so no change")
                    reference_frame = current_frame_batch.clone()
            
            adaptive_strength = color_match_strength
                
            # Set initial color_match_source if not set yet and this is the first frame
            if i == 0 and color_match_source is None:
                color_match_source = current_frame_batch.clone()
            
            # Apply adaptive color matching if enabled
            if color_match_adaptive:
                scene_change_score = 0.0
                if i > 0:
                    # Compare ORIGINAL frames for scene change detection
                    scene_change_score = self.detect_scene_change(
                        original_frames[i-1].unsqueeze(0),  # Previous ORIGINAL frame
                        current_frame_batch,                # Current ORIGINAL frame
                        color_match_scene_threshold
                    )
                elif hasattr(self, 'original_reference_buffer') and len(self.original_reference_buffer) > 0:
                    # Compare with last ORIGINAL frame from buffer
                    scene_change_score = self.detect_scene_change(
                        self.original_reference_buffer[-1].unsqueeze(0),  # Last ORIGINAL frame from buffer
                        current_frame_batch,                              # Current ORIGINAL frame
                        color_match_scene_threshold
                    )
                
                # Use threshold to determine if scene change is significant enough to reduce color matching
                if scene_change_score > color_match_scene_threshold:
                    # Calculate how much the scene change exceeds the threshold (0.0 to 1.0 range)
                    # Handle edge case where threshold is 1.0 (divisor would be 0)
                    if color_match_scene_threshold >= 1.0:
                        excess_change = 1.0  # Maximum reduction when threshold is at maximum
                    else:
                        excess_change = min((scene_change_score - color_match_scene_threshold) / (1.0 - color_match_scene_threshold), 1.0)
                    
                    # Gradually reduce strength from full strength to minimum (e.g., 20% of original)
                    min_strength_ratio = 0.2  # Don't go below 20% of original strength
                    reduction_factor = 1.0 - (excess_change * (1.0 - min_strength_ratio))
                    adaptive_strength = color_match_strength * reduction_factor
                else:
                    # Keep original strength for minor changes
                    adaptive_strength = color_match_strength
                
                print(f"Frame {i}: Scene change score: {scene_change_score:.3f}, threshold: {color_match_scene_threshold:.3f}, adaptive strength: {adaptive_strength:.3f}")
            
            # Analyze frame characteristics to choose correction method
            if adaptive_strength > 0.0:
                luminance_info = self.analyze_luminance_characteristics(reference_frame[0], current_frame)
                processed_frame = current_frame.clone()
                applied_methods = []
                
                # Calculate weights based on actual measured values (not just boolean flags)
                color_cast_strength = min(luminance_info['color_cast_check'] * 100, 1.0)  # Scale up the 0.01+ values
                contrast_strength = min(luminance_info['contrast'] / 0.25, 1.0)  # Normalize against max expected contrast
                mixed_lighting_strength = min((luminance_info['dark_region_change'] + luminance_info['bright_region_change']) * 3, 1.0)

                total_correction_needed = color_cast_strength + contrast_strength + mixed_lighting_strength
                print(f"Frame {i}: total_correction_needed: {total_correction_needed:.3f} color_cast: {color_cast_strength:.3f}, contrast: {contrast_strength:.3f}, mixed_lighting: {mixed_lighting_strength:.3f}")
                
                if total_correction_needed > requiredTotalCorrection:  # Only apply corrections if there's something significant to fix
                    # Distribute the adaptive_strength across methods based on their relative importance
                    remaining_strength = adaptive_strength
                    
                    # Step 1: White balance correction (proportional to color cast strength)
                    if color_cast_strength > 0.1 and remaining_strength > 0.05:
                        wb_weight = color_cast_strength / total_correction_needed
                        wb_strength = min(remaining_strength * wb_weight * whiteBalanceMultiply, adaptive_strength * 0.4)  # Cap at 40% of total
                        
                        if wb_strength > 0.05:
                            processed_frame = self.apply_white_balance_correction(
                                reference_frame[0], processed_frame, wb_strength * (whiteBalancePercentage / 100.0)
                            )
                            applied_methods.append(f"white_balance({wb_strength:.2f})")
                            remaining_strength -= wb_strength
                            
                            # Re-analyze after white balance to see what's left to fix
                            luminance_info_updated = self.analyze_luminance_characteristics(reference_frame[0], processed_frame)
                            # Update remaining correction needs
                            contrast_strength = min(luminance_info_updated['contrast'] / 0.25, 1.0)
                            mixed_lighting_strength = min((luminance_info_updated['dark_region_change'] + luminance_info_updated['bright_region_change']) * 15, 1.0)
                    
                    # Step 2: Luminance-preserving correction (proportional to contrast strength)
                    if contrast_strength > 0.1 and remaining_strength > 0.05:
                        lum_weight = contrast_strength / max(contrast_strength + mixed_lighting_strength, 0.1)
                        lum_strength = min(remaining_strength * lum_weight * luminanceMultiply, adaptive_strength * 0.5)  # Cap at 50% of total
                        
                        if lum_strength > 0.05:
                            processed_frame = self.apply_luminance_preserving_correction(
                                reference_frame[0], processed_frame, lum_strength * (luminancePercentage / 100.0)
                            )
                            applied_methods.append(f"luminance_preserving({lum_strength:.2f})")
                            remaining_strength -= lum_strength
                    
                    # Step 3: Zone-based correction (proportional to mixed lighting strength)
                    if mixed_lighting_strength > 0.1 and remaining_strength > 0.05:
                        zone_strength = min(remaining_strength * mixedLightingMultiply, adaptive_strength * 0.4)  # Cap at 40% of total
                        
                        if zone_strength > 0.05:
                            processed_frame = self.apply_zone_based_histogram_matching(
                                reference_frame[0], processed_frame, zone_strength * (mixedLightingPercentage / 100.0)
                            )
                            applied_methods.append(f"zone_based({zone_strength:.2f})")
                            remaining_strength -= zone_strength
                    
                    # Step 4: Fallback to regular color matching if significant strength remains unused
                    if remaining_strength > adaptive_strength * 0.01:  # If more than 1% of strength is unused
                        current_frame_batch_temp = processed_frame.unsqueeze(0)
                        color_match_result = colormatch(reference_frame, current_frame_batch_temp, color_match_method, remaining_strength)
                        processed_frame = color_match_result[0][0]
                        applied_methods.append(f"regular_colormatch({remaining_strength:.2f})")
                    
                    if applied_methods:
                        print(f"Frame {i}: Applied weighted corrections: {' -> '.join(applied_methods)}")
                    else:
                        print(f"Frame {i}: No corrections applied (strengths too low)")
                
                else:
                    # Very minor changes - fallback to original single-method logic
                    print(f"Frame {i}: Minor changes detected, using single-method fallback")
                    if luminance_info['mixed_lighting'] and luminance_info['has_high_contrast']:
                        print(f"Frame {i}: Using zone-based histogram matching (fallback)")
                        processed_frame = self.apply_zone_based_histogram_matching(
                            reference_frame[0], current_frame, adaptive_strength * 0.7
                        )
                    elif luminance_info['color_cast_detected'] and not luminance_info['has_high_contrast']:
                        print(f"Frame {i}: Using white balance correction (fallback)")
                        processed_frame = self.apply_white_balance_correction(
                            reference_frame[0], current_frame, adaptive_strength
                        )
                    elif luminance_info['has_high_contrast']:
                        print(f"Frame {i}: Using luminance-preserving correction (fallback)")
                        processed_frame = self.apply_luminance_preserving_correction(
                            reference_frame[0], current_frame, adaptive_strength
                        )
                    else:
                        print(f"Frame {i}: Using regular color matching (fallback)")
                        color_match_result = colormatch(reference_frame, current_frame_batch, color_match_method, adaptive_strength)
                        processed_frame = color_match_result[0][0]
                        
            else:
                print(f"Frame {i}: No color matching applied, adaptive strength is 0.0")
                processed_frame = current_frame
            
            # Apply temporal smoothing if enabled
            if temporal_smoothing > 0.0 and i > 0:
                processed_frame_batch = processed_frame.unsqueeze(0)
                prev_frame_batch = processed_frames[-1].unsqueeze(0)
                smoothed_frame = self.temporal_color_smooth(processed_frame_batch, prev_frame_batch, temporal_smoothing)
                processed_frame = smoothed_frame[0]
            
            processed_frames.append(processed_frame)
            
            # Update rolling buffer with CORRECTED frame for color matching (skip for firstFramesHistogram)
            if self.color_reference_method == "rolling_average":
                if len(self.color_reference_buffer) >= self.buffer_size:
                    self.color_reference_buffer.pop(0)
                self.color_reference_buffer.append(processed_frame.clone())
            
            # Also maintain original frame buffer for scene change detection (skip for firstFramesHistogram)
            if self.color_reference_method != "firstFramesHistogram":
                if not hasattr(self, 'original_reference_buffer'):
                    self.original_reference_buffer = []
                if len(self.original_reference_buffer) >= self.buffer_size:
                    self.original_reference_buffer.pop(0)
                self.original_reference_buffer.append(current_frame.clone())
        
        return torch.stack(processed_frames, dim=0)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Colour Match Advanced": VTS_ColourMatchAdvanced
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Colour Match Advanced": "VTS Colour Match Advanced"
}