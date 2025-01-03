from pathlib import Path
import pandas as pd
import os

def make_yaml():
    # Get the directory where the script is located
    script_dir = Path(__file__).parent.resolve()

    # Search for a dataset folder recursively from the script directory
    dataset_dir = next(script_dir.rglob("dataset"), None)
    if not dataset_dir:
        raise FileNotFoundError("Dataset directory not found.")
    dataset_dir = dataset_dir.resolve()

    train_path = (dataset_dir / "train/images").resolve()
    val_path = (dataset_dir / "val/images").resolve()
    test_path = (dataset_dir / "test/images").resolve()

    # Generate YAML content dynamically
    yaml_content = f'''
train: {train_path}
val: {val_path}
test: {test_path}

names:
'''

    excel_file_path = script_dir / "label_names.xlsx"
    print(f"Looking for label_names.xlsx at: {excel_file_path}")
    if not excel_file_path.is_file():
        raise FileNotFoundError("label_names.xlsx file not found.")

    data = pd.read_excel(excel_file_path)

    # Adding label names dynamically from the DataFrame
    for i, name in enumerate(data['label_names']):
        yaml_content += f"    {i}: '{name.split('.')[0]}'\n"  # Strip file extension if present

    # Append hyperparameters (optional section for augmentation and training details)
    yaml_content += '''
# Hyperparameters ------------------------------------------------------------------------------------------------------
# lr0: 0.01  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
# lrf: 0.01  # final learning rate (lr0 * lrf)
# momentum: 0.937  # SGD momentum/Adam beta1
# weight_decay: 0.0005  # optimizer weight decay 5e-4
# warmup_epochs: 3.0  # warmup epochs (fractions ok)
# warmup_momentum: 0.8  # warmup initial momentum
# warmup_bias_lr: 0.1  # warmup initial bias lr
# box: 7.5  # box loss gain
# cls: 0.5  # cls loss gain (scale with pixels)
# dfl: 1.5  # dfl loss gain
# pose: 12.0  # pose loss gain
# kobj: 1.0  # keypoint obj loss gain
# label_smoothing: 0.0  # label smoothing (fraction)
# nbs: 64  # nominal batch size
# hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
# hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
# hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.5  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.2  # image scale (+/- gain)
shear: 0.2  # image shear (+/- deg) from -0.5 to 0.5
perspective: 0.1  # image perspective (+/- fraction), range 0-0.001
flipud: 0.7  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 0.8  # image mosaic (probability)
mixup: 0.1  # image mixup (probability)
# copy_paste: 0.0  # segment copy-paste (probability)
    '''

    # Save the YAML file
    yaml_path = script_dir / 'data.yaml'
    yaml_path.write_text(yaml_content)

    print(f"YAML file created at: {yaml_path}")
    return yaml_path
