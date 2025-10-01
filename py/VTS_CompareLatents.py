import torch
import json
import numpy as np

class VTS_CompareLatents:
    """
    A node that compares two latent dictionaries and their tensor data,
    providing detailed information about differences in structure and values.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples_1": ("LATENT", ),
                "samples_2": ("LATENT", ),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("comparison_result",)
    FUNCTION = "compare_latents"
    CATEGORY = "VTS"
    
    def compare_latents(self, samples_1, samples_2):
        """
        Compare two latent dictionaries and their tensor data.
        
        Args:
            samples_1: First latent dictionary
            samples_2: Second latent dictionary
            
        Returns:
            tuple: Formatted comparison string
        """
        comparison_result = []
        comparison_result.append("=== LATENT COMPARISON REPORT ===\n")
        
        # Compare dictionary structure
        comparison_result.append("1. DICTIONARY STRUCTURE COMPARISON:")
        keys_1 = set(samples_1.keys())
        keys_2 = set(samples_2.keys())
        
        if keys_1 == keys_2:
            comparison_result.append("   ✓ Both latents have identical keys: " + str(sorted(keys_1)))
        else:
            comparison_result.append("   ✗ Different keys found:")
            comparison_result.append("     - Keys in samples_1 only: " + str(sorted(keys_1 - keys_2)))
            comparison_result.append("     - Keys in samples_2 only: " + str(sorted(keys_2 - keys_1)))
            comparison_result.append("     - Common keys: " + str(sorted(keys_1 & keys_2)))
        
        comparison_result.append("")
        
        # Compare all dictionary values
        comparison_result.append("2. DICTIONARY VALUES COMPARISON:")
        all_keys = keys_1 | keys_2
        for key in sorted(all_keys):
            if key in samples_1 and key in samples_2:
                val_1 = samples_1[key]
                val_2 = samples_2[key]
                
                if torch.is_tensor(val_1) and torch.is_tensor(val_2):
                    comparison_result.append(f"   Key '{key}': Both are tensors (detailed comparison below)")
                elif torch.is_tensor(val_1) or torch.is_tensor(val_2):
                    comparison_result.append(f"   Key '{key}': Type mismatch - one is tensor, one is not")
                    comparison_result.append(f"     - samples_1['{key}'] type: {type(val_1)}")
                    comparison_result.append(f"     - samples_2['{key}'] type: {type(val_2)}")
                else:
                    if val_1 == val_2:
                        comparison_result.append(f"   Key '{key}': ✓ Identical values - {val_1}")
                    else:
                        comparison_result.append(f"   Key '{key}': ✗ Different values")
                        comparison_result.append(f"     - samples_1['{key}']: {val_1}")
                        comparison_result.append(f"     - samples_2['{key}']: {val_2}")
            elif key in samples_1:
                comparison_result.append(f"   Key '{key}': Only in samples_1 - {samples_1[key]}")
            else:
                comparison_result.append(f"   Key '{key}': Only in samples_2 - {samples_2[key]}")
        
        comparison_result.append("")
        
        # Detailed tensor comparison for "samples" key
        comparison_result.append("3. TENSOR DATA COMPARISON (samples key):")
        
        if "samples" in samples_1 and "samples" in samples_2:
            tensor_1 = samples_1["samples"]
            tensor_2 = samples_2["samples"]
            
            if torch.is_tensor(tensor_1) and torch.is_tensor(tensor_2):
                # Shape comparison
                comparison_result.append(f"   Shape comparison:")
                comparison_result.append(f"     - samples_1 shape: {tuple(tensor_1.shape)}")
                comparison_result.append(f"     - samples_2 shape: {tuple(tensor_2.shape)}")
                
                if tensor_1.shape == tensor_2.shape:
                    comparison_result.append("     ✓ Shapes are identical")
                else:
                    comparison_result.append("     ✗ Shapes are different")
                
                # Data type comparison
                comparison_result.append(f"   Data type comparison:")
                comparison_result.append(f"     - samples_1 dtype: {tensor_1.dtype}")
                comparison_result.append(f"     - samples_2 dtype: {tensor_2.dtype}")
                
                if tensor_1.dtype == tensor_2.dtype:
                    comparison_result.append("     ✓ Data types are identical")
                else:
                    comparison_result.append("     ✗ Data types are different")
                
                # Device comparison
                comparison_result.append(f"   Device comparison:")
                comparison_result.append(f"     - samples_1 device: {tensor_1.device}")
                comparison_result.append(f"     - samples_2 device: {tensor_2.device}")
                
                # Value range comparison
                comparison_result.append(f"   Value range comparison:")
                
                with torch.no_grad():
                    min_1, max_1 = tensor_1.min().item(), tensor_1.max().item()
                    min_2, max_2 = tensor_2.min().item(), tensor_2.max().item()
                    mean_1, mean_2 = tensor_1.mean().item(), tensor_2.mean().item()
                    std_1, std_2 = tensor_1.std().item(), tensor_2.std().item()
                
                comparison_result.append(f"     - samples_1 range: [{min_1:.6f}, {max_1:.6f}]")
                comparison_result.append(f"     - samples_2 range: [{min_2:.6f}, {max_2:.6f}]")
                comparison_result.append(f"     - samples_1 mean±std: {mean_1:.6f}±{std_1:.6f}")
                comparison_result.append(f"     - samples_2 mean±std: {mean_2:.6f}±{std_2:.6f}")
                
                # Detect common ranges
                def detect_range_type(min_val, max_val):
                    if -1.1 <= min_val <= -0.9 and 0.9 <= max_val <= 1.1:
                        return "[-1, 1] (tanh-like)"
                    elif -0.1 <= min_val <= 0.1 and 0.9 <= max_val <= 1.1:
                        return "[0, 1] (sigmoid-like)"
                    elif min_val >= -0.1 and max_val <= 0.1:
                        return "Near zero (normalized)"
                    else:
                        return f"Custom range [{min_val:.3f}, {max_val:.3f}]"
                
                range_type_1 = detect_range_type(min_1, max_1)
                range_type_2 = detect_range_type(min_2, max_2)
                
                comparison_result.append(f"     - samples_1 range type: {range_type_1}")
                comparison_result.append(f"     - samples_2 range type: {range_type_2}")
                
                if range_type_1 == range_type_2:
                    comparison_result.append("     ✓ Range types are similar")
                else:
                    comparison_result.append("     ✗ Range types are different")
                
                # Tensor equality check
                comparison_result.append(f"   Tensor equality check:")
                if tensor_1.shape == tensor_2.shape:
                    try:
                        # Move tensors to same device for comparison
                        if tensor_1.device != tensor_2.device:
                            tensor_2_compare = tensor_2.to(tensor_1.device)
                        else:
                            tensor_2_compare = tensor_2
                        
                        are_equal = torch.equal(tensor_1, tensor_2_compare)
                        comparison_result.append(f"     - Tensors are identical: {are_equal}")
                        
                        if not are_equal:
                            # Calculate differences
                            with torch.no_grad():
                                diff = torch.abs(tensor_1 - tensor_2_compare)
                                max_diff = diff.max().item()
                                mean_diff = diff.mean().item()
                                
                            comparison_result.append(f"     - Maximum absolute difference: {max_diff:.8f}")
                            comparison_result.append(f"     - Mean absolute difference: {mean_diff:.8f}")
                            
                            # Check if differences are within common tolerances
                            if max_diff < 1e-6:
                                comparison_result.append("     ✓ Differences are negligible (< 1e-6)")
                            elif max_diff < 1e-4:
                                comparison_result.append("     ~ Differences are small (< 1e-4)")
                            else:
                                comparison_result.append("     ✗ Differences are significant (≥ 1e-4)")
                
                            # NEW: Conversion analysis
                            comparison_result.append("")
                            comparison_result.append("4. CONVERSION ANALYSIS:")
                            
                            # Calculate scaling factors
                            range_1 = max_1 - min_1
                            range_2 = max_2 - min_2
                            scale_factor = range_2 / range_1 if range_1 != 0 else 1.0
                            std_scale_factor = std_2 / std_1 if std_1 != 0 else 1.0
                            
                            comparison_result.append(f"   Range-based scaling:")
                            comparison_result.append(f"     - Range ratio (samples_2/samples_1): {scale_factor:.6f}")
                            comparison_result.append(f"     - Std ratio (samples_2/samples_1): {std_scale_factor:.6f}")
                            
                            # Test simple linear transformation: y = ax + b
                            # Where a = std_ratio, b = mean_difference
                            with torch.no_grad():
                                # Method 1: Direct scaling
                                tensor_1_normalized = (tensor_1 - mean_1) / std_1 * std_2 + mean_2
                                diff_normalized = torch.abs(tensor_1_normalized - tensor_2_compare)
                                max_diff_normalized = diff_normalized.max().item()
                                mean_diff_normalized = diff_normalized.mean().item()
                                
                                comparison_result.append(f"   Method 1 - Normalize to std then shift mean:")
                                comparison_result.append(f"     - Formula: (x - {mean_1:.6f}) / {std_1:.6f} * {std_2:.6f} + {mean_2:.6f}")
                                comparison_result.append(f"     - Max difference after conversion: {max_diff_normalized:.8f}")
                                comparison_result.append(f"     - Mean difference after conversion: {mean_diff_normalized:.8f}")
                                
                                if max_diff_normalized < 0.001:
                                    comparison_result.append(f"     ✓ EXCELLENT conversion (max diff < 0.001)")
                                elif max_diff_normalized < 0.01:
                                    comparison_result.append(f"     ✓ GOOD conversion (max diff < 0.01)")
                                elif max_diff_normalized < 0.1:
                                    comparison_result.append(f"     ~ OK conversion (max diff < 0.1)")
                                else:
                                    comparison_result.append(f"     ✗ Poor conversion (max diff ≥ 0.1)")
                                
                                # Method 2: Range-based scaling
                                tensor_1_range_scaled = (tensor_1 - min_1) / range_1 * range_2 + min_2
                                diff_range_scaled = torch.abs(tensor_1_range_scaled - tensor_2_compare)
                                max_diff_range_scaled = diff_range_scaled.max().item()
                                mean_diff_range_scaled = diff_range_scaled.mean().item()
                                
                                comparison_result.append(f"   Method 2 - Range-based scaling:")
                                comparison_result.append(f"     - Formula: (x - {min_1:.6f}) / {range_1:.6f} * {range_2:.6f} + {min_2:.6f}")
                                comparison_result.append(f"     - Max difference after conversion: {max_diff_range_scaled:.8f}")
                                comparison_result.append(f"     - Mean difference after conversion: {mean_diff_range_scaled:.8f}")
                                
                                if max_diff_range_scaled < 0.001:
                                    comparison_result.append(f"     ✓ EXCELLENT conversion (max diff < 0.001)")
                                elif max_diff_range_scaled < 0.01:
                                    comparison_result.append(f"     ✓ GOOD conversion (max diff < 0.01)")
                                elif max_diff_range_scaled < 0.1:
                                    comparison_result.append(f"     ~ OK conversion (max diff < 0.1)")
                                else:
                                    comparison_result.append(f"     ✗ Poor conversion (max diff ≥ 0.1)")
                                
                                # Method 3: Learned linear transformation via least squares
                                # Flatten tensors for linear regression
                                x_flat = tensor_1.flatten().float()
                                y_flat = tensor_2_compare.flatten().float()
                                
                                # Use a subset for efficiency if tensors are very large
                                n_samples = min(100000, len(x_flat))
                                if len(x_flat) > n_samples:
                                    indices = torch.randperm(len(x_flat))[:n_samples]
                                    x_subset = x_flat[indices]
                                    y_subset = y_flat[indices]
                                else:
                                    x_subset = x_flat
                                    y_subset = y_flat
                                
                                # Solve y = ax + b using least squares
                                # Stack [x, 1] for [a, b] coefficients
                                A = torch.stack([x_subset, torch.ones_like(x_subset)], dim=1)
                                try:
                                    coeffs = torch.linalg.lstsq(A, y_subset).solution
                                    a_opt, b_opt = coeffs[0].item(), coeffs[1].item()
                                    
                                    # Apply optimal transformation
                                    tensor_1_optimal = tensor_1 * a_opt + b_opt
                                    diff_optimal = torch.abs(tensor_1_optimal - tensor_2_compare)
                                    max_diff_optimal = diff_optimal.max().item()
                                    mean_diff_optimal = diff_optimal.mean().item()
                                    
                                    comparison_result.append(f"   Method 3 - Optimal linear transformation (y = ax + b):")
                                    comparison_result.append(f"     - Formula: y = {a_opt:.6f} * x + {b_opt:.6f}")
                                    comparison_result.append(f"     - Max difference after conversion: {max_diff_optimal:.8f}")
                                    comparison_result.append(f"     - Mean difference after conversion: {mean_diff_optimal:.8f}")
                                    
                                    if max_diff_optimal < 0.001:
                                        comparison_result.append(f"     ✓ EXCELLENT conversion (max diff < 0.001)")
                                    elif max_diff_optimal < 0.01:
                                        comparison_result.append(f"     ✓ GOOD conversion (max diff < 0.01)")
                                    elif max_diff_optimal < 0.1:
                                        comparison_result.append(f"     ~ OK conversion (max diff < 0.1)")
                                    else:
                                        comparison_result.append(f"     ✗ Poor conversion (max diff ≥ 0.1)")
                                        
                                except Exception as e:
                                    comparison_result.append(f"   Method 3 - Could not compute optimal transformation: {str(e)}")
                                
                                # NEW: Advanced analysis
                                comparison_result.append("")
                                comparison_result.append("5. ADVANCED ANALYSIS:")
                                
                                # Correlation analysis
                                correlation = torch.corrcoef(torch.stack([x_flat[:10000], y_flat[:10000]]))[0, 1].item()
                                comparison_result.append(f"   Correlation analysis:")
                                comparison_result.append(f"     - Pearson correlation coefficient: {correlation:.6f}")
                                
                                if abs(correlation) > 0.9:
                                    comparison_result.append("     ✓ Strong correlation - linear conversion may work with improvements")
                                elif abs(correlation) > 0.7:
                                    comparison_result.append("     ~ Moderate correlation - non-linear conversion might be needed")
                                else:
                                    comparison_result.append("     ✗ Weak correlation - may require complex conversion or be incompatible")
                                
                                # Channel-wise analysis (assuming shape [B, C, T, H, W])
                                if len(tensor_1.shape) >= 2:
                                    comparison_result.append(f"   Channel-wise analysis:")
                                    num_channels = min(tensor_1.shape[1], 8)  # Analyze first 8 channels
                                    for ch in range(num_channels):
                                        ch_1 = tensor_1[0, ch].flatten()
                                        ch_2 = tensor_2_compare[0, ch].flatten()
                                        ch_corr = torch.corrcoef(torch.stack([ch_1[:1000], ch_2[:1000]]))[0, 1].item()
                                        ch_mean_1 = ch_1.mean().item()
                                        ch_mean_2 = ch_2.mean().item()
                                        ch_std_1 = ch_1.std().item()
                                        ch_std_2 = ch_2.std().item()
                                        comparison_result.append(f"     - Channel {ch}: corr={ch_corr:.3f}, mean1={ch_mean_1:.3f}, mean2={ch_mean_2:.3f}, std1={ch_std_1:.3f}, std2={ch_std_2:.3f}")
                                
                                # Spatial pattern analysis
                                if len(tensor_1.shape) >= 4:  # Has spatial dimensions
                                    comparison_result.append(f"   Spatial pattern analysis:")
                                    # Compare spatial means across height/width
                                    spatial_mean_1 = tensor_1.mean(dim=(2, 3))  # Mean across T, H dimensions
                                    spatial_mean_2 = tensor_2_compare.mean(dim=(2, 3))
                                    spatial_corr = torch.corrcoef(torch.stack([
                                        spatial_mean_1.flatten()[:100], 
                                        spatial_mean_2.flatten()[:100]
                                    ]))[0, 1].item()
                                    comparison_result.append(f"     - Spatial mean correlation: {spatial_corr:.6f}")
                                
                                # Temporal pattern analysis (for video)
                                if len(tensor_1.shape) == 5:  # Video tensor [B, C, T, H, W]
                                    comparison_result.append(f"   Temporal pattern analysis:")
                                    temporal_mean_1 = tensor_1.mean(dim=(3, 4))  # Mean across H, W
                                    temporal_mean_2 = tensor_2_compare.mean(dim=(3, 4))
                                    temporal_corr = torch.corrcoef(torch.stack([
                                        temporal_mean_1.flatten()[:100],
                                        temporal_mean_2.flatten()[:100]
                                    ]))[0, 1].item()
                                    comparison_result.append(f"     - Temporal mean correlation: {temporal_corr:.6f}")
                                
                                # Non-linear relationship test
                                comparison_result.append(f"   Non-linear relationship test:")
                                # Test if y = a*x^2 + b*x + c fits better
                                try:
                                    x_subset_small = x_subset[:10000]
                                    y_subset_small = y_subset[:10000]
                                    A_quad = torch.stack([x_subset_small**2, x_subset_small, torch.ones_like(x_subset_small)], dim=1)
                                    coeffs_quad = torch.linalg.lstsq(A_quad, y_subset_small).solution
                                    a_quad, b_quad, c_quad = coeffs_quad[0].item(), coeffs_quad[1].item(), coeffs_quad[2].item()
                                    
                                    tensor_1_quad = a_quad * tensor_1**2 + b_quad * tensor_1 + c_quad
                                    diff_quad = torch.abs(tensor_1_quad - tensor_2_compare)
                                    max_diff_quad = diff_quad.max().item()
                                    
                                    comparison_result.append(f"     - Quadratic fit: y = {a_quad:.6f}*x² + {b_quad:.6f}*x + {c_quad:.6f}")
                                    comparison_result.append(f"     - Max difference (quadratic): {max_diff_quad:.8f}")
                                    
                                    if max_diff_quad < max_diff_optimal:
                                        improvement = ((max_diff_optimal - max_diff_quad) / max_diff_optimal) * 100
                                        comparison_result.append(f"     ✓ Quadratic fit is {improvement:.1f}% better than linear!")
                                    else:
                                        comparison_result.append(f"     - Linear fit is still better")
                                        
                                except Exception as e:
                                    comparison_result.append(f"     - Could not compute quadratic fit: {str(e)}")
                                
                                # Best method recommendation
                                comparison_result.append("")
                                methods = [
                                    ("Method 1 (Normalize)", max_diff_normalized),
                                    ("Method 2 (Range)", max_diff_range_scaled),
                                ]
                                try:
                                    methods.append(("Method 3 (Optimal Linear)", max_diff_optimal))
                                except:
                                    pass
                                try:
                                    methods.append(("Quadratic", max_diff_quad))
                                except:
                                    pass
                                
                                best_method = min(methods, key=lambda x: x[1])
                                comparison_result.append(f"   RECOMMENDATION: {best_method[0]} gives best results")
                                comparison_result.append(f"   (lowest max difference: {best_method[1]:.8f})")
                                
                                # Final assessment
                                comparison_result.append("")
                                comparison_result.append("6. CONVERSION FEASIBILITY ASSESSMENT:")
                                if best_method[1] < 0.01:
                                    comparison_result.append("   ✓ EXCELLENT: Conversion should work very well")
                                elif best_method[1] < 0.1:
                                    comparison_result.append("   ~ MODERATE: Conversion possible but may have artifacts")
                                elif best_method[1] < 1.0:
                                    comparison_result.append("   ✗ POOR: Conversion will have significant artifacts")
                                else:
                                    comparison_result.append("   ✗ INCOMPATIBLE: These latent spaces appear to be fundamentally different")
                                    comparison_result.append("   Consider using the original sampler or training a neural network converter")
                        
                    except Exception as e:
                        comparison_result.append(f"     - Could not compare tensors: {str(e)}")
                else:
                    comparison_result.append("     - Cannot compare: Different shapes")
                    
            else:
                comparison_result.append("   ✗ One or both 'samples' values are not tensors")
                comparison_result.append(f"     - samples_1['samples'] type: {type(tensor_1)}")
                comparison_result.append(f"     - samples_2['samples'] type: {type(tensor_2)}")
        else:
            if "samples" not in samples_1:
                comparison_result.append("   ✗ 'samples' key not found in samples_1")
            if "samples" not in samples_2:
                comparison_result.append("   ✗ 'samples' key not found in samples_2")
        
        # Format final result
        result_text = "\n".join(comparison_result)
        
        # Print to console for debugging
        print(result_text)
        
        return (result_text,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Compare Latents": VTS_CompareLatents
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Compare Latents": "VTS Compare Latents"
}