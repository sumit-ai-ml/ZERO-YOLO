# preprocess CT scan 
import nibabel as nib
import os 
import numpy as np
import glob 
import tqdm
def preprocess_ct_scan(ct_scan_path):
    # Load the CT scan
    ct_scan = nib.load(ct_scan_path)
    affine = ct_scan.affine
    # Get the pixel values
    ct_scan_data = ct_scan.get_fdata()
    
    # Clip the pixel values to the range [0, 80]
    ct_scan_data = np.clip(ct_scan_data, 0, 80)
    
    # save it as nifti file
    ct_scan = nib.Nifti1Image(ct_scan_data, affine)
    # save the file
    nib.save(ct_scan, ct_scan_path)