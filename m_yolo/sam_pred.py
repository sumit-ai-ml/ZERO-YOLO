# -*- encoding: utf-8 -*-
'''
@File    :   infer_with_medim_cpu.py
@Time    :   2024/09/08 11:31:02
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   Example code for inference with MedIM on CPU-only machine
'''

import medim
import torch
import numpy as np
import torch.nn.functional as F
import torchio as tio
import os.path as osp
import os
from torchio.data.io import sitk_to_nib
import SimpleITK as sitk
import requests
from tqdm import tqdm

def random_sample_next_click(prev_mask, gt_mask):
    """
    Randomly sample one click from ground-truth mask and previous seg mask

    Arguments:
        prev_mask: (torch.Tensor) [H,W,D] previous mask that SAM-Med3D predict
        gt_mask: (torch.Tensor) [H,W,D] ground-truth mask for this image
    """
    prev_mask = prev_mask > 0
    true_masks = gt_mask > 0

    if (not true_masks.any()):
        raise ValueError("Cannot find true value in the ground-truth!")

    fn_masks = torch.logical_and(true_masks, torch.logical_not(prev_mask))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), prev_mask)

    to_point_mask = torch.logical_or(fn_masks, fp_masks)

    all_points = torch.argwhere(to_point_mask)
    point = all_points[np.random.randint(len(all_points))]

    if fn_masks[point[0], point[1], point[2]]:
        is_positive = True
    else:
        is_positive = False

    sampled_point = point.clone().detach().reshape(1, 1, 3)
    sampled_label = torch.tensor([
        int(is_positive),
    ]).reshape(1, 1)

    return sampled_point, sampled_label


def sam_model_infer(model,
                    roi_image,
                    prompt_generator=random_sample_next_click,
                    roi_gt=None,
                    prev_low_res_mask=None):
    '''
    Inference for SAM-Med3D, inputs prompt points with its labels (positive/negative for each points)

    # roi_image: (torch.Tensor) cropped image, shape [1,1,128,128,128]
    # prompt_points_and_labels: (Tuple(torch.Tensor, torch.Tensor))
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device", device)
    model = model.to(device)

    with torch.no_grad():
        input_tensor = roi_image.to(device)
        image_embeddings = model.image_encoder(input_tensor)

        points_coords, points_labels = torch.zeros(1, 0,
                                                   3).to(device), torch.zeros(
                                                       1, 0).to(device)
        new_points_co, new_points_la = torch.Tensor(
            [[[64, 64, 64]]]).to(device), torch.Tensor([[1]]).to(torch.int64)
        if (roi_gt is not None):
            prev_low_res_mask = prev_low_res_mask if (
                prev_low_res_mask is not None) else torch.zeros(
                    1, 1, roi_image.shape[2] // 4, roi_image.shape[3] //
                    4, roi_image.shape[4] // 4)
            new_points_co, new_points_la = prompt_generator(
                torch.zeros_like(roi_image)[0, 0], roi_gt[0, 0])
            new_points_co, new_points_la = new_points_co.to(
                device), new_points_la.to(device)
        points_coords = torch.cat([points_coords, new_points_co], dim=1)
        points_labels = torch.cat([points_labels, new_points_la], dim=1)

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=[points_coords, points_labels],
            boxes=None,  # we currently not support bbox prompt
            masks=prev_low_res_mask.to(device),
            # masks=None,
        )

        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embeddings,  # (1, 384, 8, 8, 8)
            image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 384, 8, 8, 8)
            sparse_prompt_embeddings=sparse_embeddings,  # (1, 2, 384)
            dense_prompt_embeddings=dense_embeddings,  # (1, 384, 8, 8, 8)
        )

        prev_mask = F.interpolate(low_res_masks,
                                  size=roi_image.shape[-3:],
                                  mode='trilinear',
                                  align_corners=False)

    # convert prob to mask
    medsam_seg_prob = torch.sigmoid(prev_mask)  # (1, 1, H, W, D)
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg_mask = (medsam_seg_prob > 0.5).astype(np.uint8)

    return medsam_seg_mask


def resample_nii(input_path: str,
                 output_path: str,
                 target_spacing: tuple = (1., 1.,1.),
                 n=None,
                 reference_image=None,
                 mode="linear"):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - output_path: Path to save the resampled .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """
    # Load the nii.gz file using torchio
    subject = tio.Subject(img=tio.ScalarImage(input_path))
    resampler = tio.Resample(target=target_spacing, image_interpolation=mode)
    resampled_subject = resampler(subject)

    if (n != None):
        image = resampled_subject.img
        tensor_data = image.data
        if (isinstance(n, int)):
            n = [n]
        for ni in n:
            tensor_data[tensor_data == ni] = -1
        tensor_data[tensor_data != -1] = 0
        tensor_data[tensor_data != 0] = 1
        save_image = tio.ScalarImage(tensor=tensor_data, affine=image.affine)
        reference_size = reference_image.shape[
            1:]  # omitting the channel dimension
        cropper_or_padder = tio.CropOrPad(reference_size)
        save_image = cropper_or_padder(save_image)
    else:
        save_image = resampled_subject.img

    save_image.save(output_path)


def read_data_from_nii(img_path, gt_path):
    sitk_image = sitk.ReadImage(img_path)
    sitk_label = sitk.ReadImage(gt_path)

    if sitk_image.GetOrigin() != sitk_label.GetOrigin():
        sitk_image.SetOrigin(sitk_label.GetOrigin())
    if sitk_image.GetDirection() != sitk_label.GetDirection():
        sitk_image.SetDirection(sitk_label.GetDirection())

    sitk_image_arr, _ = sitk_to_nib(sitk_image)
    sitk_label_arr, _ = sitk_to_nib(sitk_label)

    subject = tio.Subject(
        image=tio.ScalarImage(tensor=sitk_image_arr),
        label=tio.LabelMap(tensor=sitk_label_arr),
    )
    crop_transform = tio.CropOrPad(mask_name='label',
                                   target_shape=(128, 128, 128))
    padding_params, cropping_params = crop_transform.compute_crop_or_pad(
        subject)
    if (cropping_params is None): cropping_params = (0, 0, 0, 0, 0, 0)
    if (padding_params is None): padding_params = (0, 0, 0, 0, 0, 0)

    infer_transform = tio.Compose([
        crop_transform,
        tio.ZNormalization(masking_method=lambda x: x > 0),
    ])
    subject_roi = infer_transform(subject)

    img3D_roi, gt3D_roi = subject_roi.image.data.clone().detach().unsqueeze(
        1), subject_roi.label.data.clone().detach().unsqueeze(1)
    # Update ori_roi_offset to store starting indices only
    ori_roi_offset = (
        cropping_params[0],  # Start index for the first dimension (H)
        cropping_params[2],  # Start index for the second dimension (W)
        cropping_params[4],  # Start index for the third dimension (D)
    )

    meta_info = {
        "image_path": img_path,
        "image_shape": sitk_image_arr.shape[1:],
        "origin": sitk_label.GetOrigin(),
        "direction": sitk_label.GetDirection(),
        "spacing": sitk_label.GetSpacing(),
        "padding_params": padding_params,
        "cropping_params": cropping_params,
        "ori_roi_start": ori_roi_offset,  # Update key
    }
    return (
        img3D_roi,
        gt3D_roi,
        meta_info,
    )


def save_numpy_to_nifti(in_arr: np.array, out_path, meta_info):
    # torchio turn 1xHxWxD -> DxWxH
    # so we need to squeeze and transpose back to HxWxD
    ori_arr = np.transpose(in_arr.squeeze(), (2, 1, 0))
    out = sitk.GetImageFromArray(ori_arr)
    sitk_meta_translator = lambda x: [float(i) for i in x]
    out.SetOrigin(sitk_meta_translator(meta_info["origin"]))
    out.SetDirection(sitk_meta_translator(meta_info["direction"]))
    out.SetSpacing(sitk_meta_translator(meta_info["spacing"]))
    sitk.WriteImage(out, out_path)


def data_preprocess(img_path, gt_path, category_index):
    target_img_path = osp.join(
        osp.dirname(img_path),
        osp.basename(img_path).replace(".nii.gz", "_resampled.nii.gz"))
    target_gt_path = osp.join(
        osp.dirname(gt_path),
        osp.basename(gt_path).replace(".nii.gz", "_resampled.nii.gz"))
    resample_nii(img_path, target_img_path)
    resample_nii(gt_path,
                 target_gt_path,
                 n=category_index,
                 reference_image=tio.ScalarImage(target_img_path),
                 mode="nearest")
    roi_image, roi_label, meta_info = read_data_from_nii(
        target_img_path, target_gt_path)
    return roi_image, roi_label, meta_info


def data_postprocess(roi_pred, meta_info, output_path, ori_img_path):
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    pred3D_full = np.zeros(meta_info["image_shape"])

    start_H, start_W, start_D = meta_info["ori_roi_start"]
    H_pred, W_pred, D_pred = roi_pred.shape

    end_H = start_H + H_pred
    end_W = start_W + W_pred
    end_D = start_D + D_pred

    # Ensure indices are within the bounds of pred3D_full
    end_H = min(end_H, pred3D_full.shape[0])
    end_W = min(end_W, pred3D_full.shape[1])
    end_D = min(end_D, pred3D_full.shape[2])

    pred3D_full[start_H:end_H, start_W:end_W, start_D:end_D] = roi_pred[:end_H-start_H, :end_W-start_W, :end_D-start_D]

    sitk_image = sitk.ReadImage(ori_img_path)
    ori_meta_info = {
        "image_path": ori_img_path,
        "image_shape": sitk_image.GetSize(),
        "origin": sitk_image.GetOrigin(),
        "direction": sitk_image.GetDirection(),
        "spacing": sitk_image.GetSpacing(),
    }
    pred3D_full_ori = F.interpolate(
        torch.Tensor(pred3D_full)[None][None],
        size=ori_meta_info["image_shape"],
        mode='nearest').cpu().numpy().squeeze()
    save_numpy_to_nifti(pred3D_full_ori, output_path, meta_info)

''' 1. read and pre-process your input data '''
img_path = "img1.nii.gz"
gt_path =  "img1_gt.nii.gz"
category_index = 3  # the index of your target category in the gt annotation
output_dir = "./"
roi_image, roi_label, meta_info = data_preprocess(img_path, gt_path, category_index=category_index)

''' 2. prepare the pre-trained model with local path '''
# First, download the checkpoint file locally
ckpt_url = "https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth"
ckpt_path = "./sam_med3d_turbo.pth"  # Local path to the checkpoint

# Function to download the checkpoint with progress bar
def download_checkpoint(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading checkpoint from {url}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            print("ERROR: Something went wrong during the download")
    else:
        print(f"Checkpoint already exists at {save_path}")

# Download the checkpoint if not already present
download_checkpoint(ckpt_url, ckpt_path)

# Load the checkpoint
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

# Extract the state dictionary
state_dict = checkpoint['model_state_dict']

# Create the model
model = medim.create_model("SAM-Med3D", pretrained=False)

# Load the state dict into the model
model.load_state_dict(state_dict, strict=True)

''' 3. infer with the pre-trained SAM-Med3D model '''
roi_pred = sam_model_infer(model, roi_image, roi_gt=roi_label)

''' 4. post-process and save the result '''
output_path = osp.join(output_dir, osp.basename(img_path).replace(".nii.gz", "wow_pred.nii.gz"))
data_postprocess(roi_pred, meta_info, output_path, img_path)

print("result saved to", output_path)
