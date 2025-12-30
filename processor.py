"""
Image Processor 

This module contains the core logic for transforming images by pixel
rearrangement,
"""

import numpy as np
from PIL import Image
from typing import Tuple


class ImageProcessor:
    def __init__(self):
        self.debug = True 
    
    def process(self, source: Image.Image, target: Image.Image, method: str = "simple") -> Image.Image:

        if self.debug:
            print(f"\n Processing using: {method}")
            print(f"   Source: {source.size}, Target: {target.size}")
        
        source_arr = np.array(source)
        target_arr = np.array(target)
        
        if method == "simple":
            output_arr = self._process_simple(source_arr, target_arr)
        elif method == "sorted":
            output_arr = self._process_sorted(source_arr, target_arr)
        elif method == "block":
            output_arr = self._process_block(source_arr, target_arr)
        else:
            output_arr = self._process_simple(source_arr, target_arr)

        return Image.fromarray(output_arr.astype('uint8'), 'RGB')
    
    def _process_simple(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:

        target_h, target_w = target.shape[:2]
        source_h, source_w = source.shape[:2]
        
        if (source_h, source_w) != (target_h, target_w):
            source_img = Image.fromarray(source)
            source_img = source_img.resize((target_w, target_h), Image.LANCZOS)
            source = np.array(source_img)
        
        source_flat = source.reshape(-1, 3)
        target_flat = target.reshape(-1, 3)
        
        source_brightness = source_flat.mean(axis=1)
        target_brightness = target_flat.mean(axis=1)
        
        source_order = np.argsort(source_brightness)
        target_order = np.argsort(target_brightness)
        
        output_flat = np.zeros_like(target_flat)
        
        output_flat[target_order] = source_flat[source_order]
        
        output = output_flat.reshape(target.shape)
        
        if self.debug:
            print(f"    processing complete")
            print(f"    Output shape: {output.shape}")
        
        return output
    
    def _process_sorted(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """        
        TODO: Consider local neighborhoods when sorting
        TODO: Preserve some spatial coherence from source
        """

        # For now, use simple method as baseline
        return self._process_simple(source, target)
    
    def _process_block(self, source: np.ndarray, target: np.ndarray, block_size: int = 8) -> np.ndarray:
        """        
        TODO: Implement block extraction and matching
        TODO: Handle edge cases where blocks don't align perfectly
        """

        # Block-based logic to be implemented
        if self.debug:
            print(f"   simple methode used")
        return self._process_simple(source, target)
    
    def _calculate_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:

        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        return 1.0 / (1.0 + mse) 
    
    def verify_pixel_conservation(self, source: Image.Image, output: Image.Image) -> bool:

        source_arr = np.array(source)
        output_arr = np.array(output)
        
        if source_arr.shape[:2] != output_arr.shape[:2]:
            source_img = source.resize(output.size, Image.LANCZOS)
            source_arr = np.array(source_img)
        
        source_pixels = set(map(tuple, source_arr.reshape(-1, 3)))
        output_pixels = set(map(tuple, output_arr.reshape(-1, 3)))
        
        unexpected = output_pixels - source_pixels
        
        if unexpected and self.debug:
            print(f"    Found {len(unexpected)} pixels in output not in source")
        
        return len(unexpected) == 0
