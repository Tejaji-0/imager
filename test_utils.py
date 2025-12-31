"""
Quick test script for image utilities
"""

from PIL import Image
import numpy as np
from image_utils import (
    load_image,
    resize_to_fixed_size,
    ensure_same_size,
    image_to_array,
    array_to_image,
    prepare_image_pair,
    get_image_info,
    validate_image_pair,
    DEFAULT_SIZE
)


def create_test_image(width: int, height: int, color: tuple) -> Image.Image:

    img = Image.new('RGB', (width, height), color)
    return img


def test_basic_operations():

    print("Testing image operations...")
    
    img1 = create_test_image(400, 300, (255, 0, 0))  # Red
    img2 = create_test_image(600, 400, (0, 0, 255))  # Blue
    
    print(f"Created test images: {img1.size} and {img2.size}")
    
    img1_resized = resize_to_fixed_size(img1, DEFAULT_SIZE)
    img2_resized = resize_to_fixed_size(img2, DEFAULT_SIZE)
    
    assert img1_resized.size == DEFAULT_SIZE, "Resize failed for img1"
    assert img2_resized.size == DEFAULT_SIZE, "Resize failed for img2"
    print(f"Resized both to {DEFAULT_SIZE}")

    img1_same, img2_same = ensure_same_size(img1, img2)
    assert img1_same.size == img2_same.size, "ensure_same_size failed"
    print(f"Ensured same size: {img1_same.size}")

    arr = image_to_array(img1)
    assert arr.shape == (300, 400, 3), "image_to_array shape mismatch"
    print(f"Array conversion: {arr.shape}")

    img_back = array_to_image(arr)
    assert img_back.size == img1.size, "array_to_image size mismatch"
    print(f"Array to image: {img_back.size}")


def test_image_info():
    
    print("\nTesting image info extraction...")
    
    img = create_test_image(256, 256, (128, 64, 192))
    info = get_image_info(img)
    
    assert info['width'] == 256, "Width mismatch"
    assert info['height'] == 256, "Height mismatch"
    assert info['pixel_count'] == 256 * 256, "Pixel count mismatch"
    
    print(f"Image info: {info['width']}x{info['height']}, {info['pixel_count']} pixels")
    print(f"Mean: {info['mean_value']:.2f}, Std: {info['std_value']:.2f}")


def test_validation():

    print("\nTesting image validation...")

    img1 = create_test_image(512, 512, (255, 0, 0))
    img2 = create_test_image(512, 512, (0, 255, 0))
    
    try:
        validate_image_pair(img1, img2)
        print("Validation passed for compatible images")
    except ValueError as e:
        print(f"Unexpected validation failure: {e}")
        return False

    img3 = create_test_image(256, 256, (0, 0, 255))
    
    try:
        validate_image_pair(img1, img3)
        print("Validation should have failed for incompatible images")
        return False
    except ValueError:
        print("Validation correctly rejected incompatible images")
    
    return True


def test_prepare_pair():

    print("\nTesting prepare_image_pair...")

    img1 = create_test_image(400, 300, (255, 128, 0))
    img2 = create_test_image(600, 500, (0, 128, 255))

    prep1, prep2 = prepare_image_pair(img1, img2, return_arrays=False)
    assert prep1.size == prep2.size, "Prepared images have different sizes"
    assert prep1.size == DEFAULT_SIZE, f"Size should be {DEFAULT_SIZE}"
    print(f"Prepared PIL images: {prep1.size}")

    arr1, arr2 = prepare_image_pair(img1, img2, return_arrays=True)
    assert arr1.shape == arr2.shape, "Prepared arrays have different shapes"
    print(f"Prepared arrays: {arr1.shape}")


def test_pixel_count_preservation():

    print("\nTesting pixel count preservation...")

    img1 = create_test_image(200, 150, (255, 0, 0))
    img2 = create_test_image(300, 400, (0, 255, 0))
    
    original_pixels_1 = img1.width * img1.height
    original_pixels_2 = img2.width * img2.height

    img1_resized, img2_resized = ensure_same_size(img1, img2)

    pixels_1 = img1_resized.width * img1_resized.height
    pixels_2 = img2_resized.width * img2_resized.height
    
    assert pixels_1 == pixels_2, "Pixel counts don't match after resize"
    
    print(f"Original pixel counts: {original_pixels_1:,} and {original_pixels_2:,}")
    print(f"After resize: both have {pixels_1:,} pixels")
    print(f"Both images now: {img1_resized.size}")


def run_all_tests():

    print("=" * 60)
    print("Running all tests")
    print("=" * 60)
    
    try:
        test_basic_operations()
        test_image_info()
        test_validation()
        test_prepare_pair()
        test_pixel_count_preservation()
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
