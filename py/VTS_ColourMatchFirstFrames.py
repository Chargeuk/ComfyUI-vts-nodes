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
                "editInPlace": ("BOOLEAN", {"default": False, "tooltip": "When true, modify the input image_target tensor directly instead of creating a new tensor"}),
                "gc_interval": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1, "tooltip": "Garbage collection interval. Set to 0 to disable automatic garbage collection. For large batches, lower values can help manage memory"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    DESCRIPTION = """
First Frames Histogram Color Matching

This node implements a simple and effective color matching strategy that uses the first N frames 
of a video sequence as reference frames. It builds a brightness signature library from these 
initial frames and matches subsequent frames to the most similar reference frame for color correction.

Key Features:
- Clean, focused implementation without adaptive complexity
- Histogram-based brightness matching for robust frame selection
- Consistent color correction using actual reference frames
- Ideal for maintaining temporal color stability in video sequences

The first frames are left unchanged and used to build the reference library. 
All subsequent frames are color-matched to the best-fitting reference frame.
"""

    CATEGORY = "VTS"

    def colormatch(self, image_target, method, passthrough, strength=1.0, numberOfFirstFrames=20, editInPlace=False, gc_interval=50):
        if passthrough:
            print("VTS_ColourMatchFirstFrames - passthrough is True, returning original image_target without processing")
            return (image_target,)
        
        print(f"VTS_ColourMatchFirstFrames - Processing {image_target.shape[0]} frames with first {numberOfFirstFrames} as reference library")
        
        # Initialize brightness lookup
        self.brightness_lookup = []
        self.numberOfFirstFrames = numberOfFirstFrames
        
        # Process the sequence
        output = self.process_first_frames_sequence(
            image_target,
            method,
            strength,
            editInPlace,
            gc_interval
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

    def process_first_frames_sequence(self, decoded_frames, color_match_method, color_match_strength, editInPlace, gc_interval):
        """
        Process sequence using first frames histogram matching strategy
        """
        if color_match_strength <= 0.0:
            return decoded_frames
        
        processed_frames = []
        
        for i, current_frame in enumerate(decoded_frames):
            current_frame_batch = current_frame.unsqueeze(0)  # Add batch dimension
            
            if i < self.numberOfFirstFrames:
                # Build brightness lookup from first frames (no color matching applied)
                signature = self.calculate_brightness_signature(current_frame)
                self.brightness_lookup.append({
                    'frame': current_frame.clone(),
                    'signature': signature,
                    'frame_index': i
                })
                print(f"Frame {i}: Added to brightness lookup (mean luminance: {signature['mean']:.3f}, std: {signature['std']:.3f})")
                processed_frames.append(current_frame)
            else:
                # Use brightness lookup to find best matching reference frame
                target_signature = self.calculate_brightness_signature(current_frame)
                best_match = self.find_best_brightness_match(target_signature, self.brightness_lookup)
                
                if best_match is not None:
                    reference_frame = best_match['frame'].unsqueeze(0)
                    print(f"Frame {i}: Matched with reference frame {best_match['frame_index']} (target mean lum: {target_signature['mean']:.3f})")
                    
                    # Apply color matching using the matched reference frame
                    color_match_result = colormatch(reference_frame, current_frame_batch, color_match_method, color_match_strength, editInPlace, gc_interval)
                    processed_frame = color_match_result[0][0]  # Remove batch dimension
                else:
                    print(f"Frame {i}: No suitable match found in brightness lookup, using original frame")
                    processed_frame = current_frame
                
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
