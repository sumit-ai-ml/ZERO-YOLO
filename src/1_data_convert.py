import os
import nibabel as nib
import numpy as np
from PIL import Image
import tqdm

import os
import nibabel as nib
import numpy as np
from PIL import Image
import tqdm

def save_slices_as_rgb(image_path, mask_path, output_dir, name='img', angle=0):
    # Load the 3D image and mask using nibabel
    img_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)

    # Get the image and mask data as numpy arrays
    img_data = img_nii.get_fdata()
    mask_data = mask_nii.get_fdata()

    # Ensure the image and mask have the same shape
    assert img_data.shape == mask_data.shape, "Image and mask dimensions must match."

    # Create output directories if they don’t exist
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Iterate over each slice along the z-axis (assuming it's the third axis)
    for i in tqdm.tqdm(range(img_data.shape[2])):
        img_slice = img_data[:, :, i]
        mask_slice = mask_data[:, :, i]

        # Normalize the image slice to 0-255 for saving as an 8-bit image
        if img_slice.ptp() > 0:
            img_slice = ((img_slice - img_slice.min()) / img_slice.ptp() * 255).astype(np.uint8)
        else:
            img_slice = np.zeros_like(img_slice, dtype=np.uint8)

        # Convert the mask slice to an appropriate data type
        if mask_slice.max() <= 255:
            mask_slice = mask_slice.astype(np.uint8)
            mask_mode = 'L'  # 8-bit pixels, black and white
        else:
            mask_slice = mask_slice.astype(np.uint16)
            mask_mode = 'I;16'  # 16-bit unsigned integer pixels

        # Stack the grayscale image to create an RGB image
        img_rgb = np.stack([img_slice] * 3, axis=-1)

        # Convert slices to PIL images
        img_pil = Image.fromarray(img_rgb, mode='RGB')
        mask_pil = Image.fromarray(mask_slice, mode=mask_mode)
        names = f"{name}_slice_{i}_{angle}_.tiff"

        # Save slices in the respective folders
        img_pil.save(os.path.join(images_dir, names))
        mask_pil.save(os.path.join(masks_dir, names))



'''file = os.listdir('dataset_nii/train/images')
images = 'dataset_nii/train/images'
masks = 'dataset_nii/train/labels'
output_dir = 'dataset/train'
os.makedirs(output_dir, exist_ok=True)

for i in tqdm.tqdm(range(len(file))):
    image_path = os.path.join(images, file[i])
    mask_path = os.path.join(masks, file[i])
    save_slices_as_rgb(image_path, mask_path, output_dir, name=file[i], angle=0)'''

file = os.listdir('dataset_nii/test/images')
images = 'dataset_nii/test/images'
masks = 'dataset_nii/test/labels'
output_dir = 'dataset/test'
os.makedirs(output_dir, exist_ok=True)

for i in tqdm.tqdm(range(len(file))):
    image_path = os.path.join(images, file[i])
    mask_path = os.path.join(masks, file[i])
    save_slices_as_rgb(image_path, mask_path, output_dir, name=file[i], angle=0)



