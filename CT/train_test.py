import shutil
import pandas as pd
import tqdm
import logging
import glob
import nibabel as nib
import numpy as np
import os
import glob

import os
import tqdm 
import m_yolo
from m_yolo.preprocess import process_CT_seg_data, process_CT_data

def prepare_dataset(
    meta_csv_path='meta.csv',
    mri_data_dir='MRI_data',
    output_dir='dataset_nii',
    splits=('train', 'val', 'test'),  # You can include 'test' or other splits if needed
    image_filename='mri.nii.gz',
    label_subdir='segmentations',
    label_filename='combined_mask.nii.gz',
    verbose=True
):
    """
    Prepares the dataset by organizing MRI images and their corresponding segmentation masks
    into designated training and validation directories based on the provided metadata.

    Parameters:
    - meta_csv_path (str): Path to the meta CSV file containing image IDs and split information.
    - mri_data_dir (str): Base directory where MRI data is stored.
    - output_dir (str): Base directory where the organized dataset will be saved.
    - splits (tuple): Tuple of split names to process (e.g., ('train', 'val')).
    - image_filename (str): Filename of the MRI image within each subject's directory.
    - label_subdir (str): Subdirectory name where segmentation masks are stored.
    - label_filename (str): Filename of the segmentation mask within the label subdirectory.
    - verbose (bool): If True, prints progress and information messages.

    Returns:
    - None
    """

    # Configure logging
    logging_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=logging_level, format='%(levelname)s: %(message)s')

    logging.info('Starting dataset preparation...')

    # Convert all input paths to absolute paths
    meta_csv_path = os.path.abspath(meta_csv_path)
    mri_data_dir = os.path.abspath(mri_data_dir)
    output_dir = os.path.abspath(output_dir)

    logging.info(f'Absolute path for meta CSV: {meta_csv_path}')
    logging.info(f'Absolute path for MRI data directory: {mri_data_dir}')
    logging.info(f'Absolute path for output directory: {output_dir}')

    # Step 1: Read the meta CSV file
    try:
        df = pd.read_csv(meta_csv_path)
        logging.info(f'Read meta CSV with {len(df)} entries.')
    except FileNotFoundError:
        logging.error(f'Meta CSV file not found at path: {meta_csv_path}')
        return
    except pd.errors.EmptyDataError:
        logging.error('Meta CSV file is empty.')
        return
    except Exception as e:
        logging.error(f'Error reading meta CSV: {e}')
        return

    # Validate required columns
    required_columns = {'image_id', 'split'}
    if not required_columns.issubset(df.columns):
        logging.error(f'Meta CSV must contain columns: {required_columns}')
        return

    # Step 2: Process each split
    for split in splits:
        logging.info(f'Processing split: {split}')

        # Filter dataframe for the current split
        df_split = df[df['split'] == split]
        num_entries = len(df_split)
        logging.info(f'Number of entries in "{split}": {num_entries}')

        if num_entries == 0:
            logging.warning(f'No entries found for split "{split}". Skipping.')
            continue

        # Define output subdirectories for images and labels
        images_output_dir = os.path.join(output_dir, split, 'images')
        labels_output_dir = os.path.join(output_dir, split, 'labels')

        # Create directories if they don't exist
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(labels_output_dir, exist_ok=True)
        logging.info(f'Created directories: {images_output_dir}, {labels_output_dir}')

        # Get list of image IDs
        image_ids = df_split['image_id'].values

        # Initialize counters for tracking
        success_count = 0
        missing_files = []

        # Iterate over each image ID with a progress bar
        for image_id in tqdm.tqdm(image_ids, desc=f'Processing {split}', disable=not verbose):
            # Define source paths
            image_src = os.path.join(mri_data_dir, image_id, image_filename)
            label_src = os.path.join(mri_data_dir, image_id, label_subdir, label_filename)

            # Define destination paths
            image_dst = os.path.join(images_output_dir, f'{image_id}.nii.gz')
            label_dst = os.path.join(labels_output_dir, f'{image_id}.nii.gz')

            # Check if source files exist
            if not os.path.isfile(image_src):
                logging.warning(f'Image file not found: {image_src}')
                missing_files.append(image_src)
                continue
            if not os.path.isfile(label_src):
                logging.warning(f'Label file not found: {label_src}')
                missing_files.append(label_src)
                continue

            try:
                # Copy image
                shutil.copy(image_src, image_dst)

                # Copy label
                shutil.copy(label_src, label_dst)

                success_count += 1
            except Exception as e:
                logging.error(f'Error copying files for {image_id}: {e}')
                missing_files.extend([image_src, label_src])

        logging.info(f'Completed split "{split}": {success_count}/{num_entries} files copied successfully.')
        if missing_files:
            logging.warning(f'{len(missing_files)} files were missing or failed to copy. Check warnings above.')

    logging.info('Dataset preparation completed successfully.')



import os
import glob
import tqdm

# this is to process the MRI data  related to shape and size

def ct_data():
    # Get the directory of the current Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'dataset_nii')

    # Check if the dataset directory exists
    if not os.path.exists(dataset_path):
        print(f"Error: The dataset directory {dataset_path} does not exist.")
        return

    # Use absolute paths in glob
    img_file = glob.glob(os.path.join(dataset_path, '*/images/*'))
    seg_file = glob.glob(os.path.join(dataset_path, '*/labels/*'))

    if not img_file or not seg_file:
        print("No files found in the specified directories.")
        return

    print(f"Found {len(img_file)} image files and {len(seg_file)} label files.")

    for i in tqdm.tqdm(range(len(img_file))):
        try:
            # Processing image files
            process_CT_data(img_file[i], img_file[i])
            # Processing label files
            process_CT_seg_data(seg_file[i], seg_file[i])
        except Exception as e:
            print(f"Error processing file {img_file[i]} or {seg_file[i]}: {e}")


