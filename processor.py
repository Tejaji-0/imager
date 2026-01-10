"""
Image Processor 

This module contains the core logic for transforming images by pixel
rearrangement,
"""

import numpy as np
from PIL import Image
from typing import Tuple

from image_utils import (
    prepare_image_pair,
    ensure_same_size,
    image_to_array,
    array_to_image,
    validate_image_pair,
    get_image_info,
    flatten_image,
    unflatten_image,
    verify_pixel_preservation,
    rearrange_flat_pixels
)


class ImageProcessor:
    def __init__(self):
        self.debug = True 
    
    def process(self, source: Image.Image, target: Image.Image, method: str = "simple") -> Image.Image:

        if self.debug:
            print(f"\n Processing using: {method}")
            print(f"   Source: {source.size}, Target: {target.size}")
        
        source_resized, target_resized = ensure_same_size(source, target)
        
        if self.debug and (source.size != source_resized.size or target.size != target_resized.size):
            print(f"   Resized to: {source_resized.size}")
        
        try:
            validate_image_pair(source_resized, target_resized)
        except ValueError as e:
            print(f"   Validation warning: {e}")
        
        source_arr = image_to_array(source_resized)
        target_arr = image_to_array(target_resized)
        
        if method == "simple":
            output_arr = self._process_simple(source_arr, target_arr)
        elif method == "sorted":
            output_arr = self._process_sorted(source_arr, target_arr)
        elif method == "block":
            output_arr = self._process_block(source_arr, target_arr)
        elif method == "random":
            output_arr = self._process_random(source_arr, target_arr)
        else:
            output_arr = self._process_simple(source_arr, target_arr)
        
        return array_to_image(output_arr, mode='RGB')
    
    def _process_simple(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Simple pixel-wise rearrangement using brightness-based sorting.
        
        Strategy:
        1. Both images already have same dimensions (handled by prepare step)
        2. Flatten to 1D arrays of pixels
        3. Sort all pixels from both images by brightness
        4. Map sorted source pixels to sorted target positions
        5. Reconstruct to image shape
        
        This ensures we only use source pixels (no generation)
        and create a basic resemblance to target structure.
        """
        # Flatten both images to 1D pixel arrays
        source_flat = flatten_image(source)
        target_flat = flatten_image(target)
        
        if self.debug:
            print(f"   Flattened source: {source_flat.shape}")
            print(f"   Flattened target: {target_flat.shape}")
        
        source_brightness = (
            0.2126 * source_flat[:, 0] +
            0.7152 * source_flat[:, 1] + 
            0.0722 * source_flat[:, 2]
        )
        target_brightness = (
            0.2126 * target_flat[:, 0] +
            0.7152 * target_flat[:, 1] +
            0.0722 * target_flat[:, 2]
        )
        
        source_order = np.argsort(source_brightness)
        target_order = np.argsort(target_brightness)
        
        output_flat = np.zeros_like(target_flat)
        
        output_flat[target_order] = source_flat[source_order]
        
        # Verify pixel preservation (debugging)
        if self.debug:
            source_pixels = set(map(tuple, source_flat))
            output_pixels = set(map(tuple, output_flat))
            if source_pixels == output_pixels:
                print(f"   All source pixels preserved in output")
            else:
                unexpected = output_pixels - source_pixels
                print(f"   Warning: {len(unexpected)} unexpected pixels in output")
        
        # Unflatten back to image dimensions
        height, width = target.shape[:2]
        output = unflatten_image(output_flat, height, width, 3)
        
        if self.debug:
            print(f"    Simple processing complete")
            print(f"    Output shape: {output.shape}")
            print(f"    Pixel count preserved: {source_flat.shape[0]}")
        
        return output
    
    def _process_sorted(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """        
        TODO: Consider local neighborhoods when sorting
        TODO: Preserve some spatial coherence from source
        """

        return self._process_simple(source, target)
    
    def _process_block(self, source: np.ndarray, target: np.ndarray, block_size: int = 8) -> np.ndarray:
        """        
        TODO: Implement block extraction and matching
        TODO: Handle edge cases where blocks don't align perfectly
        """

        if self.debug:
            print(f"   Block method not fully implemented yet, using simple")
        return self._process_simple(source, target)
    
    def _process_random(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:

        source_flat = flatten_image(source)
        
        if self.debug:
            print(f"   Flattened source: {source_flat.shape}")
            print(f"   Applying random permutation...")
        
        num_pixels = source_flat.shape[0]
        random_indices = np.random.permutation(num_pixels)

        rearranged_flat = rearrange_flat_pixels(source_flat, random_indices)

        if self.debug:
            source_pixels = set(map(tuple, source_flat))
            output_pixels = set(map(tuple, rearranged_flat))
            if source_pixels == output_pixels:
                print(f"   All source pixels preserved in random output")
            else:
                print(f"   Warning: Pixel preservation check failed!")

        height, width = target.shape[:2]
        output = unflatten_image(rearranged_flat, height, width, 3)
        
        if self.debug:
            print(f"   Random rearrangement complete")
            print(f"   Output shape: {output.shape}")
            print(f"   Pixels randomized: {num_pixels:,}")
        
        return output
    
    def _calculate_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:

        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        return 1.0 / (1.0 + mse) 
    
    def verify_pixel_conservation(self, source: Image.Image, output: Image.Image) -> bool:

        source_resized, output_resized = ensure_same_size(source, output)
        
        source_arr = image_to_array(source_resized)
        output_arr = image_to_array(output_resized)
        
        source_pixels = set(map(tuple, source_arr.reshape(-1, 3)))
        output_pixels = set(map(tuple, output_arr.reshape(-1, 3)))
        
        unexpected = output_pixels - source_pixels
        
        if unexpected and self.debug:
            print(f"    Found {len(unexpected)} pixels in output not in source")
        
        return len(unexpected) == 0
