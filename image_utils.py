"""
Image Loading and Processing
"""

import numpy as np
from PIL import Image
from typing import Tuple, Union, Optional
import io


DEFAULT_SIZE = (512, 512) 


def load_image(source: Union[str, bytes, Image.Image], mode: str = 'RGB') -> Image.Image:

    if isinstance(source, Image.Image):

        return source.convert(mode)
    
    elif isinstance(source, bytes):

        img = Image.open(io.BytesIO(source))
        return img.convert(mode)
    
    elif isinstance(source, str):

        img = Image.open(source)
        return img.convert(mode)
    
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")


def resize_to_fixed_size(
    img: Image.Image,
    size: Tuple[int, int] = DEFAULT_SIZE,
    resample: Image.Resampling = Image.LANCZOS
) -> Image.Image:

    if img.size == size:
        return img
    
    return img.resize(size, resample)


def ensure_same_size(
    img1: Image.Image,
    img2: Image.Image,
    target_size: Optional[Tuple[int, int]] = None,
    resample: Image.Resampling = Image.LANCZOS
) -> Tuple[Image.Image, Image.Image]:

    if target_size is None:
        target_size = DEFAULT_SIZE
    
    img1_resized = resize_to_fixed_size(img1, target_size, resample)
    img2_resized = resize_to_fixed_size(img2, target_size, resample)
    
    return img1_resized, img2_resized


def image_to_array(img: Image.Image) -> np.ndarray:
    return np.array(img)


def array_to_image(arr: np.ndarray, mode: str = 'RGB') -> Image.Image:
    arr = np.clip(arr, 0, 255).astype('uint8')
    return Image.fromarray(arr, mode)


def prepare_image_pair(
    source: Union[str, bytes, Image.Image],
    target: Union[str, bytes, Image.Image],
    size: Optional[Tuple[int, int]] = None,
    return_arrays: bool = False
) -> Union[Tuple[Image.Image, Image.Image], Tuple[np.ndarray, np.ndarray]]:

    source_img = load_image(source, mode='RGB')
    target_img = load_image(target, mode='RGB')

    source_img, target_img = ensure_same_size(source_img, target_img, size)

    if return_arrays:
        return image_to_array(source_img), image_to_array(target_img)
    
    return source_img, target_img


def get_image_info(img: Union[Image.Image, np.ndarray]) -> dict:

    if isinstance(img, Image.Image):
        arr = image_to_array(img)
        width, height = img.size
        mode = img.mode
    else:
        arr = img
        height, width = arr.shape[:2]
        channels = arr.shape[2] if arr.ndim == 3 else 1
        mode = f"{channels} channel(s)"
    
    return {
        'width': width,
        'height': height,
        'mode': mode,
        'pixel_count': width * height,
        'shape': arr.shape,
        'dtype': arr.dtype,
        'min_value': arr.min(),
        'max_value': arr.max(),
        'mean_value': arr.mean(),
        'std_value': arr.std()
    }


def validate_image_pair(
    img1: Union[Image.Image, np.ndarray],
    img2: Union[Image.Image, np.ndarray]
) -> bool:

    if isinstance(img1, Image.Image):
        shape1 = (img1.height, img1.width, len(img1.getbands()))
    else:
        shape1 = img1.shape
    
    if isinstance(img2, Image.Image):
        shape2 = (img2.height, img2.width, len(img2.getbands()))
    else:
        shape2 = img2.shape

    if shape1[:2] != shape2[:2]:
        raise ValueError(
            f"Image dimensions don't match: {shape1[:2]} vs {shape2[:2]}"
        )

    if len(shape1) != len(shape2) or shape1[2] != shape2[2]:
        raise ValueError(
            f"Channel count doesn't match: {shape1[2]} vs {shape2[2]}"
        )
    
    return True


def smart_resize(
    img: Image.Image,
    target_size: Tuple[int, int],
    maintain_aspect: bool = False
) -> Image.Image:

    if not maintain_aspect:
        return img.resize(target_size, Image.LANCZOS)

    img_copy = img.copy()
    img_copy.thumbnail(target_size, Image.LANCZOS)

    new_img = Image.new('RGB', target_size, (0, 0, 0))
    paste_x = (target_size[0] - img_copy.width) // 2
    paste_y = (target_size[1] - img_copy.height) // 2
    new_img.paste(img_copy, (paste_x, paste_y))
    
    return new_img


def resize_to_standard_sizes(img: Image.Image, size_name: str = 'medium') -> Image.Image:

    sizes = {
        'small': (256, 256),
        'medium': (512, 512),
        'large': (1024, 1024),
        'xlarge': (2048, 2048)
    }
    
    if size_name not in sizes:
        raise ValueError(f"Unknown size: {size_name}. Choose from {list(sizes.keys())}")
    
    return resize_to_fixed_size(img, sizes[size_name])


def flatten_image(img: Union[Image.Image, np.ndarray]) -> np.ndarray:

    if isinstance(img, Image.Image):
        arr = image_to_array(img)
    else:
        arr = img

    height, width = arr.shape[:2]
    num_channels = arr.shape[2] if arr.ndim == 3 else 1

    if arr.ndim == 3:
        flat = arr.reshape(-1, num_channels)
    else:
        flat = arr.reshape(-1, 1)
    
    return flat


def unflatten_image(
    flat: np.ndarray,
    height: int,
    width: int,
    channels: Optional[int] = None
) -> np.ndarray:

    if flat.ndim not in [1, 2]:
        raise ValueError(f"Flat array must be 1D or 2D, got shape {flat.shape}")

    if channels is None:
        if flat.ndim == 2:
            channels = flat.shape[1]
        else:
            channels = 1

    num_pixels = flat.shape[0]
    expected_pixels = height * width
    
    if num_pixels != expected_pixels:
        raise ValueError(
            f"Pixel count mismatch: flat array has {num_pixels} pixels, "
            f"but {height}x{width} = {expected_pixels} pixels"
        )

    if channels == 1:
        reconstructed = flat.reshape(height, width)
    else:
        reconstructed = flat.reshape(height, width, channels)
    
    return reconstructed


def verify_pixel_preservation(
    original: Union[Image.Image, np.ndarray],
    reconstructed: Union[Image.Image, np.ndarray]
) -> Tuple[bool, dict]:

    if isinstance(original, Image.Image):
        orig_arr = image_to_array(original)
    else:
        orig_arr = original
    
    if isinstance(reconstructed, Image.Image):
        recon_arr = image_to_array(reconstructed)
    else:
        recon_arr = reconstructed
    
    info = {}
    info['original_shape'] = orig_arr.shape
    info['reconstructed_shape'] = recon_arr.shape
    info['shapes_match'] = orig_arr.shape == recon_arr.shape
    
    if not info['shapes_match']:
        info['preserved'] = False
        info['reason'] = "Shape mismatch"
        return False, info

    info['pixels_equal'] = np.array_equal(orig_arr, recon_arr)

    info['original_dtype'] = str(orig_arr.dtype)
    info['reconstructed_dtype'] = str(recon_arr.dtype)
    info['dtypes_match'] = orig_arr.dtype == recon_arr.dtype

    if info['pixels_equal']:
        info['max_difference'] = 0
        info['mean_difference'] = 0.0
        info['num_different_pixels'] = 0
    else:
        diff = np.abs(orig_arr.astype(float) - recon_arr.astype(float))
        info['max_difference'] = float(diff.max())
        info['mean_difference'] = float(diff.mean())
        info['num_different_pixels'] = int(np.count_nonzero(diff))

    info['original_sum'] = int(orig_arr.sum())
    info['reconstructed_sum'] = int(recon_arr.sum())
    info['sums_match'] = info['original_sum'] == info['reconstructed_sum']

    info['preserved'] = (
        info['shapes_match'] and
        info['pixels_equal'] and
        info['dtypes_match']
    )
    
    return info['preserved'], info


def flatten_and_verify(img: Union[Image.Image, np.ndarray]) -> Tuple[np.ndarray, dict]:

    if isinstance(img, Image.Image):
        arr = image_to_array(img)
    else:
        arr = img

    height, width = arr.shape[:2]
    channels = arr.shape[2] if arr.ndim == 3 else 1

    flat = flatten_image(arr)

    reconstructed = unflatten_image(flat, height, width, channels)

    preserved, info = verify_pixel_preservation(arr, reconstructed)

    info['flattened_shape'] = flat.shape
    info['can_reconstruct'] = preserved
    
    return flat, info


def rearrange_flat_pixels(
    flat: np.ndarray,
    indices: np.ndarray
) -> np.ndarray:

    if flat.shape[0] != indices.shape[0]:
        raise ValueError(
            f"Pixel count mismatch: flat has {flat.shape[0]} pixels, "
            f"but indices has {indices.shape[0]} elements"
        )

    rearranged = flat[indices]
    
    return rearranged


__all__ = [
    'load_image',
    'resize_to_fixed_size',
    'ensure_same_size',
    'image_to_array',
    'array_to_image',
    'prepare_image_pair',
    'get_image_info',
    'validate_image_pair',
    'smart_resize',
    'resize_to_standard_sizes',
    'flatten_image',
    'unflatten_image',
    'verify_pixel_preservation',
    'flatten_and_verify',
    'rearrange_flat_pixels',
    'DEFAULT_SIZE'
]
