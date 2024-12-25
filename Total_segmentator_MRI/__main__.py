
# Import necessary modules
import os
import shutil
from pathlib import Path

# Import functions from other modules
from generate_yaml import make_yaml
from combine_mask_ import combine_mask
from train_test import prepare_dataset
from nii_to_2d import process_dataset
from tifffile_text import convert_masks_to_labels

def main():
    print("Starting the program...")

    # Call the combine_mask function
    try:
        combine_mask()
        print("Mask combination completed successfully.")
    except Exception as e:
        print(f"Error during mask combination: {e}")
        return  # Exit the program if mask combination fails

    # Get the absolute path of the script's directory
    script_dir = Path(__file__).parent.resolve()
    print(f"Script directory: {script_dir}")

    # Define relative paths based on the script's directory
    meta_csv_path = script_dir / 'meta.csv'
    mri_data_dir = script_dir / 'MRI_data'
    output_dir = script_dir / 'dataset_nii'

    # Define other parameters
    splits = ('train', 'val', 'test')  # Add 'test' or other splits if necessary
    image_filename = 'mri.nii.gz'
    label_subdir = 'segmentations'
    label_filename = 'combined_mask.nii.gz'
    verbose = True  # Set to False to reduce logging output

    # Log the absolute paths for verification
    print(f"Meta CSV absolute path: {meta_csv_path}")
    print(f"MRI data directory absolute path: {mri_data_dir}")
    print(f"Output directory absolute path: {output_dir}")

    # Call the prepare_dataset function with the defined absolute paths
    try:
        prepare_dataset(
            meta_csv_path=meta_csv_path,
            mri_data_dir=mri_data_dir,
            output_dir=output_dir,
            splits=splits,
            image_filename=image_filename,
            label_subdir=label_subdir,
            label_filename=label_filename,
            verbose=verbose
        )
        print("Dataset preparation completed successfully.")
    except Exception as e:
        print(f"Error during dataset preparation: {e}")
        return  # Exit the program if dataset preparation fails

    # Define base_dir (already resolved as script_dir)
    base_dir = script_dir

    # Define training directories
    train_images = base_dir / 'dataset_nii' / 'train' / 'images'
    train_masks = base_dir / 'dataset_nii' / 'train' / 'labels'
    train_output = base_dir / 'dataset' / 'train'

    # Rename 'test' to 'val' if 'test' exists
    print("Checking for 'test' directory to rename to 'val'...")
    src = base_dir / 'dataset_nii' / 'test'
    dst = base_dir / 'dataset_nii' / 'val'
    if src.exists() and src.is_dir():
        if dst.exists():
            print(f"'val' directory already exists at {dst}. Removing it before renaming.")
            try:
                shutil.rmtree(dst)
            except Exception as e:
                print(f"Error removing existing 'val' directory: {e}")
                return  # Exit if unable to remove existing 'val' directory
        try:
            shutil.move(str(src), str(dst))
            print(f"Renamed '{src}' to '{dst}'.")
        except Exception as e:
            print(f"Error renaming 'test' to 'val': {e}")
            return  # Exit if renaming fails
    else:
        print(f"No 'test' directory found at {src}. Skipping renaming.")

    # Define validation directories
    val_images = base_dir / 'dataset_nii' / 'val' / 'images'
    val_masks = base_dir / 'dataset_nii' / 'val' / 'labels'
    val_output = base_dir / 'dataset' / 'val'

    # Process training dataset
    print("Starting processing of training dataset...")
    try:
        process_dataset(train_images, train_masks, train_output, angle=0)
        print("Finished processing training dataset.\n")
    except Exception as e:
        print(f"Error processing training dataset: {e}")
        return  # Exit if processing fails

    # Process validation dataset
    print("Starting processing of validation dataset...")
    try:
        process_dataset(val_images, val_masks, val_output, angle=0)
        print("Finished processing validation dataset.")
    except Exception as e:
        print(f"Error processing validation dataset: {e}")
        return  # Exit if processing fails

    # Define paths for mask to label conversion
    datasets = {
        'train': {
            'input_mask_dir': base_dir / 'dataset' / 'train' / 'masks',
            'output_label_dir': base_dir / 'dataset' / 'train' / 'labels'
        },
        'val': {
            'input_mask_dir': base_dir / 'dataset' / 'val' / 'masks',
            'output_label_dir': base_dir / 'dataset' / 'val' / 'labels'
        }
    }

    # Example class mapping from 1:0 to 199999:199998
    class_mapping = {i: i - 1 for i in range(1, 200000)}

    # Iterate over each dataset split and perform conversion
    for split, paths in datasets.items():
        print(f"\nStarting conversion for '{split}' dataset:")
        try:
            convert_masks_to_labels(
                input_mask_dir=paths['input_mask_dir'],
                output_label_dir=paths['output_label_dir'],
                class_map=class_mapping
            )
            print(f"Completed conversion for '{split}' dataset.")
        except Exception as e:
            print(f"Error converting masks to labels for '{split}' dataset: {e}")
            # Decide whether to continue or exit based on the severity
            continue  # Continue with the next dataset split

    print("\nProgram completed successfully.")

    print('Generating the data.yaml file...')
    make_yaml()


if __name__ == "__main__":
    main()
