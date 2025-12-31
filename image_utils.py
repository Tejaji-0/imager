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
    'DEFAULT_SIZE'
]
