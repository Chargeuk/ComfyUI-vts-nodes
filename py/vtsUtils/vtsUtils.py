import os
import torch
from PIL import Image
import numpy as np
import comfy
import comfy.utils


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


def save_images(image_tensor, prefix="image", start_sequence=0, output_dir="./output", format="png", num_workers=4, compression_level=None, quality=None):
    """
    Save a ComfyUI image tensor to disk as lossless PNG or WebP images.
    
    Args:
        image_tensor (torch.Tensor): ComfyUI image tensor with shape (batch, height, width, channels)
        prefix (str): Prefix for the filename
        start_sequence (int): Starting sequence number
        output_dir (str): Directory to save images to
        format (str): Image format - "png" or "webp" (lossless/lossy)
        num_workers (int): Number of parallel workers for saving images (0 = sequential)
        compression_level (int): PNG compression (0-9, default 6) or WebP method (0-6, default 4 for speed)
        quality (int): For lossy WebP only (1-100, default None = lossless). PNG ignores this.
    
    Returns:
        list: List of saved file paths
    """
    from concurrent.futures import ThreadPoolExecutor
    # Validate format
    format = format.lower()
    if format not in ["png", "webp"]:
        raise ValueError(f"Unsupported format: {format}. Must be 'png' or 'webp'")
    
    # Set default compression levels for speed vs size
    if compression_level is None:
        if format == "png":
            compression_level = 6  # Default PNG compression (0=none, 9=max)
        else:  # webp
            compression_level = 4  # Default WebP method (0=fast, 6=slow/small)
    
    # Validate compression level ranges
    if format == "png" and not (0 <= compression_level <= 9):
        raise ValueError(f"PNG compression_level must be 0-9, got {compression_level}")
    if format == "webp" and not (0 <= compression_level <= 6):
        raise ValueError(f"WebP compression_level (method) must be 0-6, got {compression_level}")
    
    # Validate quality for WebP
    if quality is not None and not (1 <= quality <= 100):
        raise ValueError(f"Quality must be 1-100, got {quality}")
    
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
    number_of_images = images_np.shape[0]
    pbar = comfy.utils.ProgressBar(number_of_images)

    
    def save_single_image(args):
        """Helper function to save a single image"""
        i, image_np = args
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
        pbar.update(1)
        print(f"Saved: {filepath}")
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


class DiskImage:
    # this class represents a series of images stored to disk
    def __init__(self,
                 prefix,
                 start_sequence,
                 number_of_images,
                 output_dir,
                 format,
                 image: torch.Tensor,
                 prefetch_count=2):
        self.prefix = prefix
        self.start_sequence = start_sequence
        self.number_of_images = number_of_images
        self.output_dir = output_dir
        self.format = format
        self.shape = None
        self.dtype = None
        self.ndim = 1
        self.prefetch_count = prefetch_count
        if image is not None:
            # the provided image is likely to have a shape of B, H, W, C
            self.shape = image.shape
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
            prefetch_count=self.prefetch_count
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
        """Load a single image from disk by index"""
        if index < 0 or index >= self.number_of_images:
            raise IndexError(f"Index {index} out of range [0, {self.number_of_images})")
        
        sequence_num = self.start_sequence + index
        filename = f"{self.prefix}_{sequence_num:06d}.{self.format}"
        filepath = os.path.join(self.output_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image not found: {filepath}")
        
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
    
    def transform_and_save(self, transform_fn, batch_size=80, edit_in_place=False, new_prefix=None, new_output_dir=None):
        """
        Apply a transformation function to all images and save results.
        
        Args:
            transform_fn: Function that takes a tensor (B, H, W, C) and returns transformed tensor
            batch_size: Number of images to process at once
            edit_in_place: If True, overwrite original files. If False, create new files.
            new_prefix: Prefix for new files (required if edit_in_place=False)
            new_output_dir: Directory for new files (defaults to self.output_dir)
        
        Returns:
            DiskImage: New DiskImage object pointing to transformed images
        """
        if not edit_in_place and new_prefix is None:
            raise ValueError("new_prefix must be provided when edit_in_place=False")
        
        output_dir = new_output_dir if new_output_dir is not None else self.output_dir
        prefix = self.prefix if edit_in_place else new_prefix
        
        # Process in batches
        for i in range(0, self.number_of_images, batch_size):
            batch_count = min(batch_size, self.number_of_images - i)
            
            # Load batch
            batch_images = self.load_images(start_sequence=self.start_sequence + i, count=batch_count)
            
            # Apply transformation
            transformed = transform_fn(batch_images)
            
            # Save batch
            save_images(
                transformed,
                prefix=prefix,
                start_sequence=self.start_sequence + i,
                output_dir=output_dir,
                format=self.format
            )
        
        # Return new DiskImage
        result = DiskImage(
            prefix=prefix,
            start_sequence=self.start_sequence,
            number_of_images=self.number_of_images,
            output_dir=output_dir,
            format=self.format,
            image=None
        )
        result.shape = self.shape
        result.dtype = self.dtype
        result.ndim = self.ndim
        
        return result
