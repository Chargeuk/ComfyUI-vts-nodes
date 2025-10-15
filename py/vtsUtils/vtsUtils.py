import os
import torch
from PIL import Image
import numpy as np

# taken from comfyUi samplers.py to match the behavior of the sampler function
def get_mask_aabb(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.int)

    b = masks.shape[0]

    bounding_boxes = torch.zeros((b, 4), device=masks.device, dtype=torch.int)
    is_empty = torch.zeros((b), device=masks.device, dtype=torch.bool)
    for i in range(b):
        mask = masks[i]
        if mask.numel() == 0:
            continue
        if torch.max(mask != 0) == False:
            is_empty[i] = True
            continue
        y, x = torch.where(mask)
        bounding_boxes[i, 0] = torch.min(x)
        bounding_boxes[i, 1] = torch.min(y)
        bounding_boxes[i, 2] = torch.max(x)
        bounding_boxes[i, 3] = torch.max(y)

    return bounding_boxes, is_empty


def save_images(image_tensor, prefix="image", start_sequence=0, output_dir="./output", format="png"):
    """
    Save a ComfyUI image tensor to disk as lossless PNG or WebP images.
    
    Args:
        image_tensor (torch.Tensor): ComfyUI image tensor with shape (batch, height, width, channels)
        prefix (str): Prefix for the filename
        start_sequence (int): Starting sequence number
        output_dir (str): Directory to save images to
        format (str): Image format - "png" or "webp" (lossless)
    
    Returns:
        list: List of saved file paths
    """
    # Validate format
    format = format.lower()
    if format not in ["png", "webp"]:
        raise ValueError(f"Unsupported format: {format}. Must be 'png' or 'webp'")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensor to numpy and ensure proper data type and range
    if isinstance(image_tensor, torch.Tensor):
        # ComfyUI images are typically in range [0, 1] with shape (batch, height, width, channels)
        images_np = image_tensor.detach().cpu().numpy()
    else:
        images_np = image_tensor
    
    # Ensure values are in [0, 1] range and convert to [0, 255] uint8
    images_np = np.clip(images_np, 0.0, 1.0)
    images_np = (images_np * 255).astype(np.uint8)
    
    saved_paths = []
    
    # Save each image in the batch
    for i, image_np in enumerate(images_np):
        # Generate filename with sequence number
        sequence_num = start_sequence + i
        filename = f"{prefix}_{sequence_num:06d}.{format}"
        filepath = os.path.join(output_dir, filename)
        
        # Convert to PIL Image and save
        if image_np.shape[-1] == 3:  # RGB
            pil_image = Image.fromarray(image_np, mode='RGB')
        elif image_np.shape[-1] == 4:  # RGBA
            pil_image = Image.fromarray(image_np, mode='RGBA')
        elif image_np.shape[-1] == 1:  # Grayscale
            pil_image = Image.fromarray(image_np.squeeze(-1), mode='L')
        else:
            raise ValueError(f"Unsupported number of channels: {image_np.shape[-1]}")
        
        # Save with format-specific options
        if format == "png":
            # PNG: lossless with maximum compression
            pil_image.save(filepath, format='PNG', optimize=True)
        elif format == "webp":
            # WebP: lossless mode
            pil_image.save(filepath, format='WEBP', lossless=True, quality=100, method=6)
        
        saved_paths.append(filepath)
        print(f"Saved: {filepath}")
    
    return saved_paths

# Backward compatibility alias
def save_images_to_png(image_tensor, prefix="image", start_sequence=0, output_dir="./output"):
    """Deprecated: Use save_images() instead. This is kept for backward compatibility."""
    return save_images(image_tensor, prefix, start_sequence, output_dir, format="png")


# Example usage that would fit your VTS_Images_ScaleToMin class pattern:
def save_images_to_disk(self, image, prefix="scaled", start_sequence=0, output_dir="./output", format="png"):
    """
    Example method to save images after scaling - could be added to your VTS_Images_ScaleToMin class
    """
    saved_paths = save_images(image, prefix, start_sequence, output_dir, format)
    print(f"Saved {len(saved_paths)} images to {output_dir}")
    return saved_paths


def load_images(prefix="image", start_sequence=0, count=None, input_dir="./output", format="png"):
    """
    Load PNG or WebP images from disk into a ComfyUI image tensor.
    
    Args:
        prefix (str): Prefix of the filenames to load
        start_sequence (int): Starting sequence number
        count (int): Number of images to load. If None, loads all matching images.
        input_dir (str): Directory to load images from
        format (str): Image format - "png" or "webp"
    
    Returns:
        torch.Tensor: ComfyUI image tensor with shape (batch, height, width, channels)
    """
    # Validate format
    format = format.lower()
    if format not in ["png", "webp"]:
        raise ValueError(f"Unsupported format: {format}. Must be 'png' or 'webp'")
    
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find all matching files
    matching_files = []
    file_extension = f".{format}"
    
    if count is None:
        # Load all files matching the prefix pattern
        all_files = os.listdir(input_dir)
        pattern_prefix = f"{prefix}_"
        
        # Collect all matching files with their sequence numbers
        for filename in all_files:
            if filename.startswith(pattern_prefix) and filename.endswith(file_extension):
                matching_files.append(filename)
        
        # Sort by filename to maintain sequence order
        matching_files.sort()
    else:
        # Load specific sequence range
        for i in range(count):
            sequence_num = start_sequence + i
            filename = f"{prefix}_{sequence_num:06d}{file_extension}"
            if os.path.exists(os.path.join(input_dir, filename)):
                matching_files.append(filename)
            else:
                print(f"Warning: Expected file not found: {filename}")
    
    if not matching_files:
        raise ValueError(f"No matching images found with prefix '{prefix}' and format '{format}' in {input_dir}")
    
    print(f"Loading {len(matching_files)} {format.upper()} images from {input_dir}")
    
    # Load images
    images = []
    for filename in matching_files:
        filepath = os.path.join(input_dir, filename)
        pil_image = Image.open(filepath)
        
        # Convert to RGB if necessary (handles RGBA, L, etc.)
        if pil_image.mode == 'RGBA':
            # Keep alpha channel
            image_np = np.array(pil_image).astype(np.float32) / 255.0
        elif pil_image.mode == 'L':
            # Convert grayscale to RGB
            image_np = np.array(pil_image.convert('RGB')).astype(np.float32) / 255.0
        elif pil_image.mode == 'RGB':
            image_np = np.array(pil_image).astype(np.float32) / 255.0
        else:
            # Convert any other mode to RGB
            image_np = np.array(pil_image.convert('RGB')).astype(np.float32) / 255.0
        
        images.append(image_np)
        print(f"Loaded: {filepath}")
    
    # Stack into batch tensor
    images_tensor = torch.from_numpy(np.stack(images, axis=0))
    
    print(f"Created tensor with shape: {images_tensor.shape}")
    return images_tensor


# Backward compatibility alias
def load_images_from_png(prefix="image", start_sequence=0, count=None, input_dir="./output"):
    """Deprecated: Use load_images() instead. This is kept for backward compatibility."""
    return load_images(prefix, start_sequence, count, input_dir, format="png")


def load_images_by_pattern(pattern, input_dir="./output", sort=True):
    """
    Load PNG images from disk using a glob pattern.
    
    Args:
        pattern (str): Glob pattern to match files (e.g., "image_*.png", "frame_00*.png")
        input_dir (str): Directory to load images from
        sort (bool): Whether to sort files alphabetically before loading
    
    Returns:
        torch.Tensor: ComfyUI image tensor with shape (batch, height, width, channels)
    """
    import glob
    
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find all matching files
    search_pattern = os.path.join(input_dir, pattern)
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        raise ValueError(f"No images found matching pattern '{pattern}' in {input_dir}")
    
    if sort:
        matching_files = sorted(matching_files)
    
    print(f"Loading {len(matching_files)} images from {input_dir}")
    
    # Load images
    images = []
    for filepath in matching_files:
        pil_image = Image.open(filepath)
        
        # Convert to RGB if necessary
        if pil_image.mode == 'RGBA':
            image_np = np.array(pil_image).astype(np.float32) / 255.0
        elif pil_image.mode == 'L':
            image_np = np.array(pil_image.convert('RGB')).astype(np.float32) / 255.0
        elif pil_image.mode == 'RGB':
            image_np = np.array(pil_image).astype(np.float32) / 255.0
        else:
            image_np = np.array(pil_image.convert('RGB')).astype(np.float32) / 255.0
        
        images.append(image_np)
        print(f"Loaded: {filepath}")
    
    # Stack into batch tensor
    images_tensor = torch.from_numpy(np.stack(images, axis=0))
    
    print(f"Created tensor with shape: {images_tensor.shape}")
    return images_tensor