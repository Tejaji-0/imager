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
    flatten_image,
    unflatten_image,
    verify_pixel_preservation,
    flatten_and_verify,
    rearrange_flat_pixels,
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


def test_flatten_unflatten():

    print("\nTesting flatten and unflatten...")

    img = create_test_image(100, 80, (255, 128, 64))
    arr = image_to_array(img)
    
    original_shape = arr.shape
    print(f"Original shape: {original_shape}")

    flat = flatten_image(arr)
    expected_flat_shape = (100 * 80, 3)
    assert flat.shape == expected_flat_shape, f"Expected {expected_flat_shape}, got {flat.shape}"
    print(f"Flattened shape: {flat.shape}")

    reconstructed = unflatten_image(flat, 80, 100, 3)
    assert reconstructed.shape == original_shape, "Shape mismatch after reconstruction"
    print(f"Reconstructed shape: {reconstructed.shape}")

    assert np.array_equal(arr, reconstructed), "Pixels not preserved!"
    print(f"Pixels exactly preserved!")

    assert arr[0, 0, 0] == reconstructed[0, 0, 0], "First pixel R channel mismatch"
    assert arr[-1, -1, 2] == reconstructed[-1, -1, 2], "Last pixel B channel mismatch"
    print(f"Spot checks passed")

def test_pixel_preservation_verification():

    print("\nTesting pixel preservation verification...")

    img = create_test_image(50, 50, (200, 100, 50))
    arr = image_to_array(img)

    arr_copy = arr.copy()
    preserved, info = verify_pixel_preservation(arr, arr_copy)
    
    assert preserved, "Should detect preservation with exact copy"
    assert info['pixels_equal'], "Pixels should be equal"
    assert info['max_difference'] == 0, "Max difference should be 0"
    assert info['num_different_pixels'] == 0, "No pixels should differ"
    print(f"Correctly verified exact copy")
    print(f"  Shapes match: {info['shapes_match']}")
    print(f"  Pixels equal: {info['pixels_equal']}")
    print(f"  Max diff: {info['max_difference']}")

    arr_modified = arr.copy()
    arr_modified[0, 0, 0] = 0
    
    preserved, info = verify_pixel_preservation(arr, arr_modified)
    
    assert not preserved, "Should detect modification"
    assert not info['pixels_equal'], "Pixels should not be equal"
    assert info['num_different_pixels'] > 0, "Should detect different pixels"
    print(f"✓ Correctly detected modification")
    print(f"  Different pixels: {info['num_different_pixels']}")
    print(f"  Max diff: {info['max_difference']}")


def test_flatten_and_verify():

    print("\nTesting flatten_and_verify convenience function...")
    
    img = create_test_image(64, 64, (128, 192, 255))
    
    flat, info = flatten_and_verify(img)
    
    assert info['preserved'], "Flatten and reconstruct should preserve pixels"
    assert info['can_reconstruct'], "Should be able to reconstruct"
    assert flat.shape == (64 * 64, 3), "Flat shape incorrect"
    print(f"Flatten+verify works correctly")
    print(f"  Flattened shape: {info['flattened_shape']}")
    print(f"  Can reconstruct: {info['can_reconstruct']}")
    print(f"  Preserved: {info['preserved']}")


def test_rearrange_pixels():
    print("\nTesting pixel rearrangement...")

    width, height = 100, 80
    img_arr = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            img_arr[i, j] = [i * 2, j * 2, (i + j) % 256]

    flat = flatten_image(img_arr)
    print(f"Flattened gradient image: {flat.shape}")

    num_pixels = flat.shape[0]
    rearrange_indices = np.random.permutation(num_pixels)

    rearranged = rearrange_flat_pixels(flat, rearrange_indices)

    assert rearranged.shape == flat.shape, "Shape changed during rearrangement"
    print(f"Shape preserved: {rearranged.shape}")

    flat_set = set(map(tuple, flat))
    rearranged_set = set(map(tuple, rearranged))
    
    assert flat_set == rearranged_set, "Pixel values were modified during rearrangement!"
    print(f"All pixel values preserved (just reordered)")
    print(f"  Unique pixels: {len(flat_set)}")

    rearranged_img = unflatten_image(rearranged, height, width, 3)
    assert rearranged_img.shape == img_arr.shape, "Unflatten failed"
    print(f"Successfully unflattened to: {rearranged_img.shape}")


def test_edge_cases():

    print("\nTesting edge cases...")
    
    gray_img = create_test_image(50, 50, (128, 128, 128)).convert('L')
    gray_arr = np.array(gray_img)
    
    flat_gray = flatten_image(gray_arr)
    assert flat_gray.shape == (50 * 50, 1), "Grayscale flatten failed"
    print(f"Grayscale flattening works: {flat_gray.shape}")
    
    reconstructed_gray = unflatten_image(flat_gray, 50, 50, 1)
    assert np.array_equal(gray_arr, reconstructed_gray.squeeze()), "Grayscale reconstruction failed"
    print(f"Grayscale reconstruction works")
    
    try:
        wrong_flat = np.zeros((100, 3))
        unflatten_image(wrong_flat, 10, 11, 3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly caught pixel count mismatch")

    pil_img = create_test_image(40, 30, (255, 0, 0))
    flat_pil = flatten_image(pil_img)
    assert flat_pil.shape == (40 * 30, 3), "PIL Image flatten failed"
    print(f"PIL Image input works: {flat_pil.shape}")


def test_real_world_workflow():

    source = create_test_image(200, 150, (255, 100, 50))
    target = create_test_image(180, 160, (50, 150, 200))
    
    print(f"Source: {source.size}, Target: {target.size}")

    source_resized, target_resized = ensure_same_size(source, target)
    print(f"Resized both to: {source_resized.size}")

    source_arr = image_to_array(source_resized)
    target_arr = image_to_array(target_resized)

    source_flat = flatten_image(source_arr)
    target_flat = flatten_image(target_arr)
    
    assert source_flat.shape == target_flat.shape, "Flattened shapes don't match"
    print(f"Both flattened to: {source_flat.shape}")

    source_brightness = source_flat.mean(axis=1)
    target_brightness = target_flat.mean(axis=1)
    
    source_order = np.argsort(source_brightness)
    target_order = np.argsort(target_brightness)

    output_flat = np.zeros_like(target_flat)
    output_flat[target_order] = source_flat[source_order]

    source_pixel_set = set(map(tuple, source_flat))
    output_pixel_set = set(map(tuple, output_flat))
    
    assert source_pixel_set == output_pixel_set, "Pixels were modified!"
    print(f"Rearrangement preserved all source pixels")

    h, w = source_arr.shape[:2]
    output_arr = unflatten_image(output_flat, h, w, 3)
    output_img = array_to_image(output_arr)
    
    assert output_img.size == source_resized.size, "Output size mismatch"
    print(f"Successfully reconstructed to: {output_img.size}")
    print(f"Complete workflow successful!")


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
        test_flatten_unflatten()
        test_pixel_preservation_verification()
        test_flatten_and_verify()
        test_rearrange_pixels()
        test_edge_cases()
        test_real_world_workflow()
        
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
