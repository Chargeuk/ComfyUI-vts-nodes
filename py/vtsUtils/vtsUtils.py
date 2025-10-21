import os
import torch
from PIL import Image
import numpy as np
import comfy
import comfy.utils
import time

vtsImageTypes = ["jpg", "webp", "png"]
vtsReturnTypes = ["Input", "DiskImage", "Tensor"]


def get_default_image_input_types(prefix="image"):
    return {
            "required": {
                "return_type": (vtsReturnTypes, {"default": vtsReturnTypes[0]}),
                "image": ("IMAGE", {"default": None }),
                "batch_size": ("INT", {"default": 20, "min": 1}),
                "edit_in_place": ("BOOLEAN", {"default": False}),
                "prefix": ("STRING", {"default": prefix, "multiline": False}),
                "start_sequence": ("INT", {"default": 0, "min": 0}),
                "output_dir": ("STRING", {"default": "./tmp/images", "multiline": False}),
                "format": (vtsImageTypes, {"default": vtsImageTypes[0]}),
                "num_workers": ("INT", {"default": 16, "min": 1}),
                "compression_level": ("INT", {"default": 9, "min": 0, "max": 9, "tooltip": "Image compression level (0-9 for png and 0-6 for WebP)"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 101, "tooltip": "Image quality (1-100), or 101 for lossless. Only affects WebP"}),
            }
        }

def ensure_image_defaults(image_data: dict) -> dict:
    """
    Ensure that all required fields are present in the image data dictionary.
    If a field is missing, it will be added with a default value.

    Args:
        image_data: The image data dictionary to check

    Returns:
        The updated image data dictionary with defaults applied
    """
    # first, get the return type
    return_type = image_data.get("return_type", None)
    if return_type is None or return_type == "Input":
        image = image_data.get("image", None)
        if image is not None and not isinstance(image, torch.Tensor):
            # we have to make all fields equal to the input image's fields
            image_data["prefix"] = image.prefix
            image_data["start_sequence"] = image.start_sequence
            image_data["output_dir"] = image.output_dir
            image_data["format"] = image.format
            image_data["compression_level"] = image.compression_level
            image_data["quality"] = image.quality

    defaults = get_default_image_input_types()
    for key, value in defaults["required"].items():
        if key not in image_data or image_data[key] is None:
            image_data[key] = value[1]["default"]

    quality = image_data.get("quality", 95)
    if quality > 100:
        image_data["quality"] = None
    return image_data

def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries. Values from override will overwrite values in base.
    
    Args:
        base: The base dictionary
        override: Dictionary with values to merge/overwrite
        
    Returns:
        A new merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Overwrite with the new value
            result[key] = value
    
    return result

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


def save_images(
        image,
        prefix="image",
        start_sequence=0,
        output_dir="./output",
        format="png",
        num_workers=4,
        compression_level=None,
        quality=None,
        max_retries=5,
        **kwargs, # Accept and ignore extra kwargs
    ):
    """
    Save a ComfyUI image tensor to disk as PNG, WebP, or JPG images.
    
    Args:
        image (torch.Tensor): ComfyUI image tensor with shape (batch, height, width, channels)
        prefix (str): Prefix for the filename
        start_sequence (int): Starting sequence number
        output_dir (str): Directory to save images to
        format (str): Image format - "png", "webp", "jpg", or "jpeg"
        num_workers (int): Number of parallel workers for saving images (0 = sequential)
        compression_level (int): PNG compression (0-9, default 6) or WebP method (0-6, default 4 for speed). Ignored for JPG.
        quality (int): For lossy WebP (1-100, default None = lossless) or JPG (1-100, default 95). PNG ignores this.
        max_retries (int): Maximum number of retry attempts for file save operations (default 5)
    
    Returns:
        list: List of saved file paths
    """
    from concurrent.futures import ThreadPoolExecutor
    # Validate format
    format = format.lower()
    if format == "jpeg":
        format = "jpg"  # Normalize jpeg to jpg
    if format not in vtsImageTypes:
        raise ValueError(f"Unsupported format: {format}. Must be 'png', 'webp', 'jpg', or 'jpeg'")
    
    # Set default compression levels for speed vs size
    if compression_level is None:
        if format == "png":
            compression_level = 6  # Default PNG compression (0=none, 9=max)
        elif format == "webp":
            compression_level = 4  # Default WebP method (0=fast, 6=slow/small)
        # JPG doesn't use compression_level

    if compression_level < 0:
        print(f"Warning: compression_level < 0 ({compression_level}), clamping to 0")
        compression_level = 0
    
    # Set default quality
    if quality is None:
        if format == "jpg":
            quality = 95  # Default JPG quality (1-100, higher is better)
        # PNG doesn't use quality, WebP defaults to lossless (None)
    
    # Validate compression level ranges
    if format == "png" and not (compression_level <= 9):
        print(f"Warning: PNG compression_level > 9 ({compression_level}), clamping to 9")
        compression_level = 9
    if format == "webp" and compression_level is not None and not (compression_level <= 6):
        print(f"Warning: WebP compression_level (method) must be 0-6, got {compression_level}, clamping to 6")
        compression_level = 6  # Default WebP method (0=fast, 6=slow/small)

    # Validate quality for WebP and JPG
    if quality is not None and not (1 <= quality <= 100):
        print(f"Warning: Quality must be 1-100, got {quality}. defaulting to 95")
        quality = 95  # Default JPG quality

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensor to numpy and ensure proper data type and range
    if isinstance(image, torch.Tensor):
        # ComfyUI images are typically in range [0, 1] with shape (batch, height, width, channels)
        images_np = image.detach().cpu().numpy()
    else:
        images_np = image
    
    # Ensure values are in [0, 1] range and convert to [0, 255] uint8
    images_np = np.clip(images_np, 0.0, 1.0)
    images_np = (images_np * 255).astype(np.uint8)
    number_of_images = images_np.shape[0]
    pbar = comfy.utils.ProgressBar(number_of_images)

    
    def save_single_image(args):
        """Helper function to save a single image with retry logic"""
        i, image_np = args
        sequence_num = start_sequence + i
        filename = f"{prefix}_{sequence_num:06d}.{format}"
        filepath = os.path.join(output_dir, filename)

        # Convert to PIL Image
        if image_np.shape[-1] == 3:  # RGB
            pil_image = Image.fromarray(image_np, mode='RGB')
        elif image_np.shape[-1] == 4:  # RGBA
            # JPG doesn't support alpha channel, convert to RGB
            if format == "jpg":
                # Convert RGBA to RGB by compositing on white background
                image_rgba = Image.fromarray(image_np, mode='RGBA')
                background = Image.new('RGB', image_rgba.size, (255, 255, 255))
                background.paste(image_rgba, mask=image_rgba.split()[3])  # Use alpha channel as mask
                pil_image = background
            else:
                pil_image = Image.fromarray(image_np, mode='RGBA')
        elif image_np.shape[-1] == 1:  # Grayscale
            pil_image = Image.fromarray(image_np.squeeze(-1), mode='L')
        else:
            raise ValueError(f"Unsupported number of channels: {image_np.shape[-1]}")

        # Save with retry logic and exponential backoff
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Save with format-specific options
                if format == "png":
                    # PNG compression levels: 0=no compression (fast/large), 9=max compression (slow/small)
                    # optimize=True enables additional optimization passes
                    pil_image.save(
                        filepath, 
                        format='PNG', 
                        compress_level=compression_level,
                        optimize=(compression_level > 0)
                    )
                elif format == "webp":
                    # WebP can be lossless or lossy based on if quality is set from 0 to 100 or is none
                    if quality is None:
                        # Lossless WebP
                        pil_image.save(
                            filepath, 
                            format='WEBP', 
                            lossless=True, 
                            quality=100,
                            method=compression_level
                        )
                    else:
                        # Lossy WebP (much smaller files)
                        pil_image.save(
                            filepath, 
                            format='WEBP', 
                            lossless=False, 
                            quality=quality,
                            method=compression_level
                        )
                elif format == "jpg":
                    # JPG is always lossy, quality 1-100 (higher is better)
                    # optimize=True enables additional optimization
                    pil_image.save(
                        filepath,
                        format='JPEG',
                        quality=quality,
                        optimize=True
                    )
                
                # If save succeeded, break out of retry loop
                pbar.update(1)
                print(f"Saved: {filepath} compression_level={compression_level} quality={quality}")
                return filepath
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 0.1s, 0.2s, 0.4s, 0.8s, 1.6s
                    wait_time = 0.1 * (2 ** attempt)
                    print(f"Warning: Failed to save {filepath} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed, raise the exception
                    print(f"Error: Failed to save {filepath} after {max_retries} attempts: {e}")
                    raise
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        return filepath
    
    # Prepare arguments for parallel processing
    save_args = list(enumerate(images_np))
    
    # Save images in parallel if num_workers > 0
    if num_workers > 0 and len(save_args) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            saved_paths = list(executor.map(save_single_image, save_args))
    else:
        # Sequential processing
        saved_paths = [save_single_image(args) for args in save_args]
    
    return saved_paths


def save_images_async(
        image,
        prefix="image",
        start_sequence=0,
        output_dir="./output",
        format="png",
        num_workers=4,
        compression_level=None,
        quality=None,
        max_retries=5,
        **kwargs,
    ):
    """
    Save a ComfyUI image tensor to disk asynchronously in a background thread.
    Returns immediately with a Future object that can be checked or waited on.
    
    Args:
        image (torch.Tensor): ComfyUI image tensor with shape (batch, height, width, channels)
        prefix (str): Prefix for the filename
        start_sequence (int): Starting sequence number
        output_dir (str): Directory to save images to
        format (str): Image format - "png", "webp", "jpg", or "jpeg"
        num_workers (int): Number of parallel workers for saving images (0 = sequential)
        compression_level (int): PNG compression (0-9, default 6) or WebP method (0-6, default 4 for speed). Ignored for JPG.
        quality (int): For lossy WebP (1-100, default None = lossless) or JPG (1-100, default 95). PNG ignores this.
        max_retries (int): Maximum number of retry attempts for file save operations (default 5)
    
    Returns:
        concurrent.futures.Future: A Future object representing the save operation.
                                   Call .result() to wait for completion and get the list of saved file paths.
                                   Call .done() to check if the operation is complete without blocking.
                                   Call .cancel() to attempt to cancel the operation.
    
    Example:
        # Start saving in background
        future = save_images_async(images, prefix="frame", output_dir="./output")
        
        # Do other work...
        
        # Check if done (non-blocking)
        if future.done():
            print("Save complete!")
        
        # Wait for completion and get results
        saved_paths = future.result()
        print(f"Saved {len(saved_paths)} images")
    """
    from concurrent.futures import ThreadPoolExecutor
    
    # Create a single-thread executor for the background save operation
    executor = ThreadPoolExecutor(max_workers=1)
    
    # Submit the save_images function to run in background
    future = executor.submit(
        save_images,
        image=image,
        prefix=prefix,
        start_sequence=start_sequence,
        output_dir=output_dir,
        format=format,
        num_workers=num_workers,
        compression_level=compression_level,
        quality=quality,
        max_retries=max_retries,
        **kwargs
    )
    
    # Attach cleanup callback to shutdown executor when done
    def cleanup(f):
        executor.shutdown(wait=False)
    
    future.add_done_callback(cleanup)
    
    return future


def load_images(prefix="image", start_sequence=0, count=None, input_dir="./output", format="png", num_workers=4, max_retries=5):
    """
    Load PNG, WebP, or JPG images from disk into a ComfyUI image tensor.
    Uses parallel loading for improved performance.
    
    Args:
        prefix (str): Prefix of the filenames to load
        start_sequence (int): Starting sequence number
        count (int): Number of images to load. If None, loads all matching images.
        input_dir (str): Directory to load images from
        format (str): Image format - "png", "webp", "jpg", or "jpeg"
        num_workers (int): Number of parallel workers for loading images (default 4)
        max_retries (int): Maximum number of retry attempts for file load operations (default 5)
    
    Returns:
        torch.Tensor: ComfyUI image tensor with shape (batch, height, width, channels)
    """
    from concurrent.futures import ThreadPoolExecutor
    
    # Validate format
    format = format.lower()
    if format == "jpeg":
        format = "jpg"  # Normalize jpeg to jpg
    if format not in vtsImageTypes:
        raise ValueError(f"Unsupported format: {format}. Must be 'png', 'webp', 'jpg', or 'jpeg'")
    
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
    
    def load_single_image(filename):
        """Helper function to load a single image with retry logic"""
        filepath = os.path.join(input_dir, filename)
        
        # Load with retry logic and exponential backoff
        last_exception = None
        for attempt in range(max_retries):
            try:
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
                
                # If load succeeded, break out of retry loop
                print(f"Loaded: {filepath}")
                return image_np
                
            except (OSError, IOError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 0.1s, 0.2s, 0.4s, 0.8s, 1.6s
                    wait_time = 0.1 * (2 ** attempt)
                    print(f"Warning: Failed to load {filepath} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed, raise the exception
                    print(f"Error: Failed to load {filepath} after {max_retries} attempts: {e}")
                    raise
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        return image_np
    
    # Load images in parallel
    if num_workers > 0 and len(matching_files) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            images = list(executor.map(load_single_image, matching_files))
    else:
        # Sequential loading
        images = [load_single_image(filename) for filename in matching_files]
    
    # Stack into batch tensor
    images_tensor = torch.from_numpy(np.stack(images, axis=0))
    
    print(f"Created tensor with shape: {images_tensor.shape}")
    return images_tensor


def load_images_async(prefix="image", start_sequence=0, count=None, input_dir="./output", format="png", num_workers=4, max_retries=5):
    """
    Load PNG, WebP, or JPG images from disk asynchronously in a background thread.
    Returns immediately with a Future object that can be checked or waited on.
    
    Args:
        prefix (str): Prefix of the filenames to load
        start_sequence (int): Starting sequence number
        count (int): Number of images to load. If None, loads all matching images.
        input_dir (str): Directory to load images from
        format (str): Image format - "png", "webp", "jpg", or "jpeg"
        num_workers (int): Number of parallel workers for loading images (default 4)
        max_retries (int): Maximum number of retry attempts for file load operations (default 5)
    
    Returns:
        concurrent.futures.Future: A Future object representing the load operation.
                                   Call .result() to wait for completion and get the image tensor.
                                   Call .done() to check if the operation is complete without blocking.
                                   Call .cancel() to attempt to cancel the operation.
    
    Example:
        # Start loading in background
        future = load_images_async(prefix="frame", input_dir="./output", count=100)
        
        # Do other work...
        
        # Check if done (non-blocking)
        if future.done():
            print("Load complete!")
        
        # Wait for completion and get results
        images_tensor = future.result()
        print(f"Loaded tensor with shape: {images_tensor.shape}")
    """
    from concurrent.futures import ThreadPoolExecutor
    
    # Create a single-thread executor for the background load operation
    executor = ThreadPoolExecutor(max_workers=1)
    
    # Submit the load_images function to run in background
    future = executor.submit(
        load_images,
        prefix=prefix,
        start_sequence=start_sequence,
        count=count,
        input_dir=input_dir,
        format=format,
        num_workers=num_workers,
        max_retries=max_retries
    )
    
    # Attach cleanup callback to shutdown executor when done
    def cleanup(f):
        executor.shutdown(wait=False)
    
    future.add_done_callback(cleanup)
    
    return future


def load_images_by_pattern(pattern, input_dir="./output", sort=True, max_retries=5):
    """
    Load PNG images from disk using a glob pattern.
    
    Args:
        pattern (str): Glob pattern to match files (e.g., "image_*.png", "frame_00*.png")
        input_dir (str): Directory to load images from
        sort (bool): Whether to sort files alphabetically before loading
        max_retries (int): Maximum number of retry attempts for file load operations (default 5)
    
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
    
    # Load images with retry logic
    images = []
    for filepath in matching_files:
        # Load with retry logic and exponential backoff
        last_exception = None
        for attempt in range(max_retries):
            try:
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
                
                # If load succeeded, break out of retry loop
                images.append(image_np)
                print(f"Loaded: {filepath}")
                break
                
            except (OSError, IOError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 0.1s, 0.2s, 0.4s, 0.8s, 1.6s
                    wait_time = 0.1 * (2 ** attempt)
                    print(f"Warning: Failed to load {filepath} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed, raise the exception
                    print(f"Error: Failed to load {filepath} after {max_retries} attempts: {e}")
                    raise
    
    # Stack into batch tensor
    images_tensor = torch.from_numpy(np.stack(images, axis=0))
    
    print(f"Created tensor with shape: {images_tensor.shape}")
    return images_tensor


def transform_images(
    images,
    transform_fn,
    batch_size=80,
    num_workers=8
):
    """
    Apply a transformation function to images without saving or returning results.
    Uses a pipeline approach with parallel loading for optimal performance.
    The transform_fn is called but its return value is discarded.
    
    Args:
        images (DiskImage or torch.Tensor): DiskImage instance or tensor with shape (batch, height, width, channels)
        transform_fn: Function that takes a tensor (B, H, W, C) and performs some operation.
                     Return value is ignored.
        batch_size: Number of images to process at once
        num_workers: Number of parallel workers for background loading (default 8)
    
    Returns:
        None
    """
    from concurrent.futures import ThreadPoolExecutor
    
    # Determine if input is a tensor or DiskImage
    is_tensor = isinstance(images, torch.Tensor)
    
    if is_tensor:
        # Extract metadata from tensor
        number_of_images = images.shape[0]
        
        # Create a simple loader function for tensor batches
        def load_batch(batch_start, batch_count):
            batch_end = min(batch_start + batch_count, number_of_images)
            return images[batch_start:batch_end]
            
    else:
        # Handle DiskImage input
        number_of_images = images.number_of_images
        
        # Create loader function for DiskImage
        def load_batch(batch_start, batch_count):
            return images.load_images(start_sequence=images.start_sequence + batch_start, count=batch_count)
    
    # Calculate number of batches
    num_batches = (number_of_images + batch_size - 1) // batch_size
    
    # Use executor for background loading
    with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
        # Prefetch first batch
        next_batch_future = None
        batch_num = 0
        
        for batch_idx in range(0, number_of_images, batch_size):
            batch_count = min(batch_size, number_of_images - batch_idx)
            batch_num += 1
            
            # If we don't have a prefetch in progress, load synchronously (first iteration)
            if next_batch_future is None:
                print(f"Loading batch {batch_num}/{num_batches}")
                batch_images = load_batch(batch_idx, batch_count)
            else:
                # Wait for the prefetched batch to finish loading
                print(f"Waiting for prefetched batch {batch_num}/{num_batches}")
                batch_images = next_batch_future.result()
            
            # Start loading next batch in background (if there is one)
            next_batch_idx = batch_idx + batch_size
            if next_batch_idx < number_of_images:
                next_batch_count = min(batch_size, number_of_images - next_batch_idx)
                print(f"Prefetching batch {batch_num + 1}/{num_batches} in background")
                next_batch_future = executor.submit(
                    load_batch,
                    next_batch_idx,
                    next_batch_count
                )
            else:
                next_batch_future = None
            
            # Transform current batch (while next batch loads in background)
            print(f"Transforming batch {batch_num}/{num_batches}")
            transform_fn(batch_images)
            
            # Clean up current batch
            del batch_images
    
    print(f"Transform complete: processed {number_of_images} images")


def transform_and_save_images(
    image,
    transform_fn,
    batch_size=80,
    edit_in_place=False,
    prefix=None,
    output_dir=None,
    num_workers=16,
    format=None,
    return_type=None,
    compression_level=None,
    quality=None,
    **kwargs, # Accept and ignore extra kwargs
):
    """
    Apply a transformation function to images and save results.
    Uses a pipeline approach with parallel loading and saving for optimal performance.
    
    Args:
        image (DiskImage or torch.Tensor): DiskImage instance or tensor with shape (batch, height, width, channels)
        transform_fn: Function that takes a tensor (B, H, W, C) and returns transformed tensor
                     Can return different number of images than input.
        batch_size: Number of images to process at once
        edit_in_place: If True, overwrite original files (only for DiskImage) or modify input tensor in-place.
                      For tensors: requires transform_fn to return same batch size as input, else raises ValueError.
        prefix: Prefix for new files (required if edit_in_place=False or images is a tensor and return_type is DiskImage)
        output_dir: Directory for new files (required if images is a tensor and return_type is DiskImage, defaults to images.output_dir for DiskImage)
        num_workers: Number of parallel workers for saving images (default 16)
        format: Image format for output (default "png", only used if images is a tensor)
        return_type: Return type - "DiskImage", "Tensor", "Input", or None (default None)
                    If None, returns the same type as the input images.
                    If "Tensor", loads all transformed images into memory and returns as tensor (no disk writes).
                    If "DiskImage", saves to disk and returns a DiskImage object pointing to saved files.
    
    Returns:
        DiskImage or torch.Tensor: Depending on return_type parameter (or input type if return_type is None)
    """
    from concurrent.futures import ThreadPoolExecutor
    
    # Determine if input is a tensor or DiskImage
    is_tensor = isinstance(image, torch.Tensor)
    
    # If return_type is None, default to the same type as input
    if return_type is None or return_type == "Input":
        return_type = "Tensor" if is_tensor else "DiskImage"

    # Validate return_type
    if return_type not in ["DiskImage", "Tensor"]:
        raise ValueError(f"return_type must be 'DiskImage', 'Tensor', or None, got '{return_type}'")

    if is_tensor:
        # Handle tensor input
        if return_type == "DiskImage":
            if prefix is None:
                raise ValueError("prefix must be provided when images is a tensor and return_type is DiskImage")
            if output_dir is None:
                raise ValueError("output_dir must be provided when images is a tensor and return_type is DiskImage")

        # Extract metadata from tensor
        number_of_images = image.shape[0]
        original_shape = image.shape
        original_dtype = image.dtype
        original_ndim = image.ndim
        input_format = format
        if input_format is None:
            input_format = vtsImageTypes[0]  # Default to first item in vtsImageTypes
        if prefix is None:
            prefix = "tmp"
        output_dir = output_dir if output_dir else "./temp"
        start_sequence = 0
        
        # Create a simple loader function for tensor batches
        def load_batch(batch_start, batch_count):
            batch_end = min(batch_start + batch_count, number_of_images)
            return image[batch_start:batch_end]
            
    else:
        # Handle DiskImage input
        if return_type == "DiskImage" and not edit_in_place and prefix is None:
            raise ValueError("prefix must be provided when edit_in_place=False")
        
        number_of_images = image.number_of_images
        original_shape = image.shape
        original_dtype = image.dtype
        original_ndim = image.ndim if image.ndim is not None else 1
        input_format = format
        if edit_in_place or input_format is None:
            input_format = image.format
        if input_format is None:
            input_format = vtsImageTypes[0]  # Default to first item in vtsImageTypes
        output_dir = output_dir if output_dir is not None else image.output_dir
        prefix = image.prefix if edit_in_place else (prefix if prefix else "temp")
        start_sequence = image.start_sequence

        if compression_level is None:
            # Apply compression level to the image saving process
            compression_level = image.compression_level

        if quality is None:
            quality = image.quality

        # Create loader function for DiskImage
        def load_batch(batch_start, batch_count):
            return image.load_images(start_sequence=image.start_sequence + batch_start, count=batch_count)

    # Calculate number of batches
    num_batches = (number_of_images + batch_size - 1) // batch_size
    
    # Track total output images and current output sequence
    total_output_images = 0
    current_output_sequence = start_sequence if edit_in_place else 0
    output_shape = None
    output_dtype = None
    
    # For tensor return type, collect all transformed batches (only if not editing in place)
    transformed_batches = [] if (return_type == "Tensor" and not (is_tensor and edit_in_place)) else None
    
    # Use executor for background loading and saving
    with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
        # Prefetch first batch
        next_batch_future = None
        batch_num = 0
        save_futures = []  # Track save operations to ensure completion
        
        for batch_idx in range(0, number_of_images, batch_size):
            batch_count = min(batch_size, number_of_images - batch_idx)
            batch_num += 1
            
            # If we don't have a prefetch in progress, load synchronously (first iteration)
            if next_batch_future is None:
                print(f"Loading batch {batch_num}/{num_batches}")
                batch_images = load_batch(batch_idx, batch_count)
            else:
                # Wait for the prefetched batch to finish loading
                print(f"Waiting for prefetched batch {batch_num}/{num_batches}")
                batch_images = next_batch_future.result()
            
            # Start loading next batch in background (if there is one)
            next_batch_idx = batch_idx + batch_size
            if next_batch_idx < number_of_images:
                next_batch_count = min(batch_size, number_of_images - next_batch_idx)
                print(f"Prefetching batch {batch_num + 1}/{num_batches} in background")
                next_batch_future = executor.submit(
                    load_batch,
                    next_batch_idx,
                    next_batch_count
                )
            else:
                next_batch_future = None
            
            # Transform current batch (while next batch loads in background)
            print(f"Transforming batch {batch_num}/{num_batches}")
            transformed = transform_fn(batch_images)
            
            # Determine number of output images from this batch
            if isinstance(transformed, torch.Tensor):
                num_output = transformed.shape[0]
                if output_shape is None:
                    output_shape = (None,) + transformed.shape[1:]  # Store H, W, C
                    output_dtype = transformed.dtype
            else:
                num_output = len(transformed)
            
            print(f"Batch {batch_num}/{num_batches}: {batch_count} input -> {num_output} output")
            
            # Handle based on return type
            if return_type == "Tensor":
                # Check if we're doing in-place editing on a tensor
                if is_tensor and edit_in_place:
                    # Verify batch size matches
                    if num_output != batch_count:
                        raise ValueError(
                            f"edit_in_place=True requires transform_fn to return same batch size as input. "
                            f"Expected {batch_count} images, got {num_output} images in batch {batch_num}/{num_batches}"
                        )
                    
                    # Copy transformed batch back into original tensor
                    batch_end = batch_idx + batch_count
                    if isinstance(transformed, torch.Tensor):
                        image[batch_idx:batch_end] = transformed.to(image.device)
                    else:
                        image[batch_idx:batch_end] = torch.tensor(transformed, device=image.device)

                    print(f"Updated input tensor in-place: indices {batch_idx} to {batch_end-1}")
                else:
                    # For tensor return without edit_in_place, collect the transformed batch (no disk writes)
                    if isinstance(transformed, torch.Tensor):
                        transformed_batches.append(transformed.cpu())
                    else:
                        transformed_batches.append(transformed)
            else:
                # For DiskImage return, save to disk
                print(f"Saving batch {batch_num}/{num_batches} in background (sequence {current_output_sequence} to {current_output_sequence + num_output - 1})")
                save_future = executor.submit(
                    save_images,
                    transformed.cpu() if hasattr(transformed, 'cpu') else transformed,
                    prefix=prefix,
                    start_sequence=current_output_sequence,
                    output_dir=output_dir,
                    format=input_format,
                    num_workers=num_workers,
                    compression_level=compression_level,
                    quality=quality
                )
                save_futures.append(save_future)
            
            # Update counters
            total_output_images += num_output
            current_output_sequence += num_output
            
            # Clean up current batch
            del batch_images
            if return_type != "Tensor" or (is_tensor and edit_in_place):
                del transformed
        
        # Wait for all saves to complete (only if saving to disk)
        if return_type == "DiskImage":
            print("Waiting for all save operations to complete...")
            for future in save_futures:
                future.result()
    
    # Return based on return_type
    if return_type == "Tensor":
        # If editing in place, return the modified input tensor
        if is_tensor and edit_in_place:
            print(f"Transform complete: {number_of_images} images transformed in-place")
            return image

        # Otherwise, concatenate all batches into a single tensor
        if transformed_batches:
            if isinstance(transformed_batches[0], torch.Tensor):
                result_tensor = torch.cat(transformed_batches, dim=0)
            else:
                # If not tensors, try to stack as numpy then convert
                result_tensor = torch.from_numpy(np.concatenate(transformed_batches, axis=0))
            
            print(f"Transform complete: {number_of_images} input images -> {total_output_images} output images (returned as tensor)")
            return result_tensor
        else:
            raise RuntimeError("No transformed batches were collected")
    
    else:  # return_type == "DiskImage"
        # Prepare result metadata
        result_shape = None
        result_dtype = None
        result_ndim = original_ndim
        
        if output_shape is not None:
            result_shape = (total_output_images,) + output_shape[1:]
            result_dtype = output_dtype
        else:
            result_shape = original_shape
            result_dtype = original_dtype
        
        print(f"Transform complete: {number_of_images} input images -> {total_output_images} output images")
        
        # Create and return new DiskImage
        result = DiskImage(
            prefix=prefix,
            start_sequence=start_sequence if edit_in_place else 0,
            number_of_images=total_output_images,
            output_dir=output_dir,
            format=input_format,
            image=None,
            compression_level=compression_level,
            quality=quality
        )
        result.shape = result_shape
        result.dtype = result_dtype
        result.ndim = result_ndim
        
        return result


class DiskImage:
    # this class represents a series of images stored to disk
    def __init__(self,
                 prefix,
                 start_sequence,
                 number_of_images,
                 output_dir,
                 format,
                 image: torch.Tensor,
                 prefetch_count=2,
                 compression_level=None,
                 quality=None,
                 **kwargs): # Accept and ignore extra kwargs
        self.prefix = prefix
        self.start_sequence = start_sequence
        self.number_of_images = number_of_images
        self.output_dir = output_dir
        self.format = format
        self.shape = None
        self.dtype = None
        self.ndim = 1
        self.prefetch_count = prefetch_count
        self.compression_level = compression_level
        self.quality = quality
        if image is not None:
            # the provided image is likely to have a shape of B, H, W, C
            self.shape = image.shape
            if self.number_of_images is None:
                self.number_of_images = self.shape[0]

            # Create new shape tuple with updated batch size
            self.shape = (self.number_of_images,) + tuple(self.shape[1:])
            self.dtype = image.dtype
            self.ndim = image.ndim

    def clone(self):
        selfCopy =  DiskImage(
            prefix=self.prefix,
            start_sequence=self.start_sequence,
            number_of_images=self.number_of_images,
            output_dir=self.output_dir,
            format=self.format,
            image=None,
            prefetch_count=self.prefetch_count,
            compression_level=self.compression_level,
            quality=self.quality
        )
        selfCopy.shape = self.shape
        selfCopy.dtype = self.dtype
        selfCopy.ndim = self.ndim
        return selfCopy

    def detach(self):
        return self.clone()

    def to(self, *args, device=None, dtype=None, **kwargs):
        # Ignore device and dtype since DiskImage stores on disk
        return self.clone()

    def len(self):
        numberOfImages = self.number_of_images
        return numberOfImages
    
    def __len__(self):
        return self.number_of_images

    def __getitem__(self, index):
        """Load a single image from disk by index with retry logic"""
        if index < 0 or index >= self.number_of_images:
            raise IndexError(f"Index {index} out of range [0, {self.number_of_images})")
        
        sequence_num = self.start_sequence + index
        filename = f"{self.prefix}_{sequence_num:06d}.{self.format}"
        filepath = os.path.join(self.output_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image not found: {filepath}")
        
        # Load with retry logic and exponential backoff
        max_retries = 5
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Load image and convert to tensor matching expected format
                pil_image = Image.open(filepath)
                
                # Convert to numpy array in [0, 1] range
                if pil_image.mode == 'RGBA':
                    image_np = np.array(pil_image).astype(np.float32) / 255.0
                elif pil_image.mode in ['L', 'P']:
                    image_np = np.array(pil_image.convert('RGB')).astype(np.float32) / 255.0
                elif pil_image.mode == 'RGB':
                    image_np = np.array(pil_image).astype(np.float32) / 255.0
                else:
                    image_np = np.array(pil_image.convert('RGB')).astype(np.float32) / 255.0
                
                # Convert to torch tensor with shape (H, W, C)
                return torch.from_numpy(image_np)
                
            except (OSError, IOError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 0.1s, 0.2s, 0.4s, 0.8s, 1.6s
                    wait_time = 0.1 * (2 ** attempt)
                    print(f"Warning: Failed to load {filepath} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed, raise the exception
                    print(f"Error: Failed to load {filepath} after {max_retries} attempts: {e}")
                    raise
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception

    def __iter__(self):
        """Make DiskImage iterable for use in VideoCombine"""
        return self.iter_with_prefetch(prefetch_count=self.prefetch_count)

    def iter_with_prefetch(self, prefetch_count=None):
        """
        Alternative iterator with configurable prefetch count.
        Allows per-iteration override of the default prefetch_count.
        
        Args:
            prefetch_count (int): Number of images to prefetch ahead. 
                                 If None, uses self.prefetch_count
        
        Yields:
            torch.Tensor: Images loaded from disk
        """
        from concurrent.futures import ThreadPoolExecutor
        
        if prefetch_count is None:
            prefetch_count = self.prefetch_count
        
        if self.number_of_images == 0:
            return
        
        with ThreadPoolExecutor(max_workers=prefetch_count) as executor:
            # Submit initial batch of prefetch tasks
            futures = {}
            for i in range(min(prefetch_count, self.number_of_images)):
                futures[i] = executor.submit(self.__getitem__, i)
            
            # Yield images and submit new prefetch tasks
            for i in range(self.number_of_images):
                # Get the current image (will wait if not ready)
                image = futures[i].result()
                
                # Submit next prefetch task
                next_idx = i + prefetch_count
                if next_idx < self.number_of_images:
                    futures[next_idx] = executor.submit(self.__getitem__, next_idx)
                
                # Clean up used future
                del futures[i]
                
                yield image

    def load_images(self, start_sequence=0, count=None):
        result = load_images(prefix=self.prefix, start_sequence=start_sequence, count=count, input_dir=self.output_dir, format=self.format)
        return result
    
    def transform_and_save(self, transform_fn, batch_size=80, edit_in_place=False, new_prefix=None, new_output_dir=None, num_workers=16, return_type=None):
        """
        Apply a transformation function to all images and save results.
        Uses a pipeline approach with parallel loading and saving for optimal performance.
        
        Args:
            transform_fn: Function that takes a tensor (B, H, W, C) and returns transformed tensor
                        Can return different number of images than input.
            batch_size: Number of images to process at once
            edit_in_place: If True, overwrite original files. If False, create new files.
            new_prefix: Prefix for new files (required if edit_in_place=False and return_type is DiskImage)
            new_output_dir: Directory for new files (defaults to self.output_dir)
            num_workers: Number of parallel workers for saving images (default 16)
            return_type: Return type - "DiskImage", "Tensor", or None (default None)
                        If None, returns DiskImage (same as input type).
        
        Returns:
            DiskImage or torch.Tensor: Depending on return_type parameter
        """
        # Call the standalone function and return the result directly
        return transform_and_save_images(
            image=self,
            transform_fn=transform_fn,
            batch_size=batch_size,
            edit_in_place=edit_in_place,
            prefix=new_prefix,
            output_dir=new_output_dir,
            num_workers=num_workers,
            format=self.format,
            return_type=return_type
        )



