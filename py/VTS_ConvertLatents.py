import torch
import json
import numpy as np

class VTS_ConvertLatents:
    """
    A node that converts latents from one sampler format to another using 
    channel-wise linear transformations based on statistical analysis.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_latent": ("LATENT", ),
                "conversion_method": (["channel_wise", "global_linear", "global_normalize"], {"default": "channel_wise"}),
            },
            "optional": {
                "reference_source": ("LATENT", {"tooltip": "Reference latent from source sampler (for learning conversion)"}),
                "reference_target": ("LATENT", {"tooltip": "Reference latent from target sampler (for learning conversion)"}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("converted_latent", "conversion_info")
    FUNCTION = "convert_latents"
    CATEGORY = "VTS"
    
    def convert_latents(self, source_latent, conversion_method, reference_source=None, reference_target=None):
        """
        Convert latents using learned or predefined transformations.
        """
        result_info = []
        result_info.append("=== LATENT CONVERSION REPORT ===\n")
        
        # Copy the source latent structure
        converted_latent = source_latent.copy()
        
        if "samples" not in source_latent:
            error_msg = "Error: 'samples' key not found in source latent"
            result_info.append(error_msg)
            return (source_latent, "\n".join(result_info))
        
        source_tensor = source_latent["samples"]
        
        # If reference latents are provided, learn the conversion
        if reference_source is not None and reference_target is not None:
            result_info.append("Learning conversion from reference latents...")
            
            if "samples" not in reference_source or "samples" not in reference_target:
                error_msg = "Error: Reference latents missing 'samples' key"
                result_info.append(error_msg)
                return (source_latent, "\n".join(result_info))
            
            ref_source_tensor = reference_source["samples"]
            ref_target_tensor = reference_target["samples"]
            
            # Move to same device for computation
            if ref_source_tensor.device != ref_target_tensor.device:
                ref_target_tensor = ref_target_tensor.to(ref_source_tensor.device)
            
            if ref_source_tensor.shape != ref_target_tensor.shape:
                error_msg = f"Error: Reference tensor shapes don't match: {ref_source_tensor.shape} vs {ref_target_tensor.shape}"
                result_info.append(error_msg)
                return (source_latent, "\n".join(result_info))
            
            with torch.no_grad():
                if conversion_method == "channel_wise":
                    result_info.append("Using channel-wise linear conversion...")
                    converted_tensor = self._convert_channel_wise(source_tensor, ref_source_tensor, ref_target_tensor, result_info)
                    
                elif conversion_method == "global_linear":
                    result_info.append("Using global linear conversion...")
                    converted_tensor = self._convert_global_linear(source_tensor, ref_source_tensor, ref_target_tensor, result_info)
                    
                elif conversion_method == "global_normalize":
                    result_info.append("Using global normalization conversion...")
                    converted_tensor = self._convert_global_normalize(source_tensor, ref_source_tensor, ref_target_tensor, result_info)
                    
                else:
                    error_msg = f"Error: Unknown conversion method: {conversion_method}"
                    result_info.append(error_msg)
                    return (source_latent, "\n".join(result_info))
            
        else:
            # Use predefined conversion based on your analysis results
            result_info.append("Using predefined conversion parameters...")
            result_info.append("(Based on wan video model sampler analysis)")
            
            with torch.no_grad():
                if conversion_method == "channel_wise":
                    converted_tensor = self._convert_channel_wise_predefined(source_tensor, result_info)
                elif conversion_method == "global_normalize":
                    converted_tensor = self._convert_global_normalize_predefined(source_tensor, result_info)
                else:
                    # Fallback to global normalize
                    converted_tensor = self._convert_global_normalize_predefined(source_tensor, result_info)
        
        # Update the converted latent
        converted_latent["samples"] = converted_tensor
        
        # Add noise_mask if it doesn't exist (based on your analysis)
        if "noise_mask" not in converted_latent:
            converted_latent["noise_mask"] = None
            result_info.append("Added 'noise_mask': None to match target format")
        
        result_info.append(f"\nConversion completed successfully!")
        result_info.append(f"Output shape: {tuple(converted_tensor.shape)}")
        result_info.append(f"Output range: [{converted_tensor.min().item():.6f}, {converted_tensor.max().item():.6f}]")
        result_info.append(f"Output mean±std: {converted_tensor.mean().item():.6f}±{converted_tensor.std().item():.6f}")
        
        return (converted_latent, "\n".join(result_info))
    
    def _convert_channel_wise(self, source_tensor, ref_source, ref_target, result_info):
        """Convert using per-channel linear transformations."""
        converted = source_tensor.clone()
        
        # Assuming shape [B, C, T, H, W] or [B, C, H, W]
        num_channels = source_tensor.shape[1]
        
        for ch in range(num_channels):
            # Get channel data
            source_ch = source_tensor[:, ch]
            ref_source_ch = ref_source[:, ch]
            ref_target_ch = ref_target[:, ch]
            
            # Calculate transformation parameters for this channel
            with torch.no_grad():
                # Flatten for statistics
                ref_source_flat = ref_source_ch.flatten().float()
                ref_target_flat = ref_target_ch.flatten().float()
                
                # Use subset for efficiency
                n_samples = min(10000, len(ref_source_flat))
                if len(ref_source_flat) > n_samples:
                    indices = torch.randperm(len(ref_source_flat))[:n_samples]
                    x_subset = ref_source_flat[indices]
                    y_subset = ref_target_flat[indices]
                else:
                    x_subset = ref_source_flat
                    y_subset = ref_target_flat
                
                # Solve y = ax + b for this channel
                A = torch.stack([x_subset, torch.ones_like(x_subset)], dim=1)
                try:
                    coeffs = torch.linalg.lstsq(A, y_subset).solution
                    a, b = coeffs[0].item(), coeffs[1].item()
                    
                    # Apply transformation
                    converted[:, ch] = source_ch * a + b
                    
                    result_info.append(f"  Channel {ch}: y = {a:.6f} * x + {b:.6f}")
                    
                except Exception as e:
                    # Fallback to statistical normalization
                    mean_src = ref_source_ch.mean()
                    std_src = ref_source_ch.std()
                    mean_tgt = ref_target_ch.mean()
                    std_tgt = ref_target_ch.std()
                    
                    converted[:, ch] = (source_ch - mean_src) / std_src * std_tgt + mean_tgt
                    result_info.append(f"  Channel {ch}: Statistical normalization (error: {str(e)})")
        
        return converted
    
    def _convert_global_linear(self, source_tensor, ref_source, ref_target, result_info):
        """Convert using global linear transformation."""
        # Flatten tensors
        ref_source_flat = ref_source.flatten().float()
        ref_target_flat = ref_target.flatten().float()
        
        # Use subset for efficiency
        n_samples = min(100000, len(ref_source_flat))
        if len(ref_source_flat) > n_samples:
            indices = torch.randperm(len(ref_source_flat))[:n_samples]
            x_subset = ref_source_flat[indices]
            y_subset = ref_target_flat[indices]
        else:
            x_subset = ref_source_flat
            y_subset = ref_target_flat
        
        # Solve y = ax + b
        A = torch.stack([x_subset, torch.ones_like(x_subset)], dim=1)
        coeffs = torch.linalg.lstsq(A, y_subset).solution
        a, b = coeffs[0].item(), coeffs[1].item()
        
        result_info.append(f"  Global linear: y = {a:.6f} * x + {b:.6f}")
        
        return source_tensor * a + b
    
    def _convert_global_normalize(self, source_tensor, ref_source, ref_target, result_info):
        """Convert using global statistical normalization."""
        mean_src = ref_source.mean()
        std_src = ref_source.std()
        mean_tgt = ref_target.mean()
        std_tgt = ref_target.std()
        
        result_info.append(f"  Global normalize: (x - {mean_src:.6f}) / {std_src:.6f} * {std_tgt:.6f} + {mean_tgt:.6f}")
        
        return (source_tensor - mean_src) / std_src * std_tgt + mean_tgt
    
    def _convert_channel_wise_predefined(self, source_tensor, result_info):
        """Convert using predefined channel-wise parameters from analysis."""
        converted = source_tensor.clone()
        
        # Based on learned coefficients from successful wan video model conversion
        # These provide visually identical results to the target sampler
        channel_params = [
            # [a, b] for y = ax + b transformation per channel
            (0.345731, 0.266601),   # Channel 0
            (0.668525, 0.514007),   # Channel 1
            (0.424411, 0.389445),   # Channel 2
            (0.373739, -0.033028),  # Channel 3
            (0.822565, 0.159338),   # Channel 4
            (0.552002, -0.529137),  # Channel 5
            (0.374092, 0.064140),   # Channel 6
            (0.484338, -0.763688),  # Channel 7
            (0.300309, -0.120046),  # Channel 8
            (0.465372, 0.066264),   # Channel 9
            (0.339786, -0.189196),  # Channel 10
            (0.617863, 0.242260),   # Channel 11
            (0.608520, 0.123674),   # Channel 12
            (0.876538, 0.838254),   # Channel 13
            (0.352272, -0.079095),  # Channel 14
            (0.524633, 0.163276),   # Channel 15
        ]
        
        num_channels = min(source_tensor.shape[1], len(channel_params))
        
        for ch in range(num_channels):
            a, b = channel_params[ch]
            converted[:, ch] = source_tensor[:, ch] * a + b
            result_info.append(f"  Channel {ch}: y = {a:.6f} * x + {b:.6f}")
        
        # For any additional channels beyond 16, use the last learned parameters as fallback
        if source_tensor.shape[1] > len(channel_params):
            fallback_a, fallback_b = 0.524633, 0.163276  # Channel 15 parameters
            for ch in range(len(channel_params), source_tensor.shape[1]):
                converted[:, ch] = source_tensor[:, ch] * fallback_a + fallback_b
                result_info.append(f"  Channel {ch}: y = {fallback_a:.6f} * x + {fallback_b:.6f} (fallback)")
        
        return converted
    
    def _convert_global_normalize_predefined(self, source_tensor, result_info):
        """Convert using predefined global normalization from analysis."""
        # From your analysis: Method 1 (Normalize) was best
        mean_src = -0.224886
        std_src = 1.868329
        mean_tgt = -0.084378
        std_tgt = 0.752230
        
        result_info.append(f"  Global normalize: (x - {mean_src:.6f}) / {std_src:.6f} * {std_tgt:.6f} + {mean_tgt:.6f}")
        
        return (source_tensor - mean_src) / std_src * std_tgt + mean_tgt

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "VTS Convert Latents": VTS_ConvertLatents
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Convert Latents": "VTS Convert Latents"
}