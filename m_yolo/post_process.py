import numpy as np

def reshape_to_match(original_image, other_image):
    """
    Reshape `other_image` to match the shape of `original_image` by:
    1. Permuting axes based on matching dimension sizes.
    2. Slicing or padding each dimension to match the target shape.

    Parameters:
    - original_image (numpy.ndarray): Reference image with target shape.
    - other_image (numpy.ndarray): Image to be reshaped.

    Returns:
    - numpy.ndarray: Reshaped `other_image` with the same shape as `original_image`.
    """
    original_shape = original_image.shape
    other_shape = other_image.shape

    # Step 1: Permute axes to match original_image's axes based on size
    permutation = []
    used_axes = set()

    for o_size in original_shape:
        # Find axis in other_shape that matches the current original dimension size
        matched_axis = None
        for i, size in enumerate(other_shape):
            if size == o_size and i not in used_axes:
                matched_axis = i
                break
        if matched_axis is not None:
            permutation.append(matched_axis)
            used_axes.add(matched_axis)
        else:
            # If exact match not found, find the closest size
            closest_axis = min(
                [(i, abs(o_size - size)) for i, size in enumerate(other_shape) if i not in used_axes],
                key=lambda x: x[1],
                default=(None, None)
            )
            if closest_axis[0] is not None:
                permutation.append(closest_axis[0])
                used_axes.add(closest_axis[0])
            else:
                raise ValueError("Cannot find a suitable axis to match.")

    # Permute the other_image axes
    try:
        permuted_other = other_image.transpose(permutation)
    except Exception as e:
        raise ValueError(f"Error in transposing axes: {e}")

    # Step 2: Initialize the reshaped image with zeros
    reshaped = np.zeros(original_shape, dtype=permuted_other.dtype)

    # Step 3: Determine slicing indices for each dimension
    slices_original = []
    slices_other = []
    for o_dim, p_dim in zip(original_shape, permuted_other.shape):
        min_dim = min(o_dim, p_dim)
        slices_original.append(slice(0, min_dim))
        slices_other.append(slice(0, min_dim))

    # Step 4: Assign the overlapping region from permuted_other to reshaped
    reshaped[tuple(slices_original)] = permuted_other[tuple(slices_other)]

    return reshaped

'''i = 11
# Example usage
final, _ = get_data(final_file[i])
original, _ = get_data(original_file[i])
print(final.shape, original.shape)
print(final_file[i])

reshaped_image = reshape_to_match(original, final)

print("Original image shape:", original.shape)
print("Other image shape (before):", final.shape)
print("Other image shape (after):", reshaped_image.shape)

# save reshaped_image as nii image 

nib.save(nib.Nifti1Image(reshaped_image, _), 'output_file.nii.gz')'''