---
title: 'ZERO-YOLO: A No-Code GUI for Training Custom YOLO Segmentation Models'
tags:
  - deep learning
  - image segmentation
  - YOLO
  - Streamlit
  - no-code
  - computer vision
  - open source
authors:
  - name: Sumit Pandey
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Satyasaran Changdar
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Erik Bjørnager Dam
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Department of Computer Science, University of Copenhagen
    index: 1
date: 2025-06-03
repository: https://github.com/sumit-ai-ml/ZERO-YOLO

---

date: 2025-06-03
repository: https://github.com/sumit-ai-ml/ZERO-YOLO
---

# Summary

Image segmentation is central to diverse domains, from healthcare to agriculture, yet remains out of reach for many due to the programming expertise required. **ZERO-YOLO** addresses this gap by offering a truly no-code graphical user interface (GUI) for the entire workflow of training and evaluating YOLO-based segmentation models. Built on Streamlit, ZERO-YOLO empowers researchers, educators, and domain experts—regardless of coding background—to harness the power of modern deep learning for segmentation tasks.

# Statement of Need

Deep-learning models such as YOLO (You Only Look Once) have transformed object detection and segmentation, but their complexity often restricts adoption to those with advanced technical skills. Many scientists and professionals need custom segmentation tools but cannot invest in the steep learning curve of coding, data formatting, and model training. **ZERO-YOLO** provides an accessible, interactive interface, lowering barriers for non-coders and democratizing access to high-performance segmentation across fields like medicine, ecology, and industry.

# Features

**ZERO-YOLO** enables the full YOLO segmentation pipeline within a single unified GUI. Key features include:

- **Data Preparation & Harmonization**
  - Converts images and masks from `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` into standardized `.tiff` format.
  - Ensures masks use correct pixel values per class and filenames are synchronized.
- **Automated Dataset Splitting**
  - Customizable train/validation/test splits (default 70/15/15).
- **YOLO-Compatible Annotation Conversion**
  - Segmentation masks auto-converted to YOLO `.txt` annotations.
  - Configurable class mapping via GUI.
- **YAML Configuration Generation**
  - Automated creation of YOLO YAML config files.
  - Upload class labels from Excel.
  - Adjust augmentation and hyperparameters: rotation, scaling, mosaic, mix-up, etc.
- **YOLO Training Pipeline**
  - Direct integration with Ultralytics YOLOv8.
  - Configurable epochs, batch size, image size, device (GPU/CPU), layer freezing.
- **Interactive Visualization Playground**
  - Real-time visualization: original images, predicted masks, overlays, prediction confidence.
- **Mask Generation & Evaluation**
  - Automated inference to generate masks from new images.
  - Built-in metrics: Dice Score, Intersection-over-Union (IoU), sensitivity, specificity.

# Domain Impact

ZERO-YOLO democratizes access to advanced segmentation, benefiting a wide spectrum of domains:

- **Healthcare:** Medical image segmentation (e.g., CT, MRI) without coding.
- **Agriculture:** Crop monitoring, pest detection from field/drone imagery.
- **Environmental Science:** Segmentation of satellite/drone imagery for ecological studies.
- **Education:** Enables hands-on teaching of deep-learning segmentation without programming barriers.

# State of the Field

Most YOLO and segmentation tools require command-line or programming knowledge, limiting accessibility for non-technical users. ZERO-YOLO distinguishes itself with a fully graphical, no-code interface covering the complete workflow—from data harmonization and annotation conversion to training, inference, and performance evaluation.

# Illustrative Examples

- **Medical Imaging:** Clinicians can segment anatomical structures from CT or MRI scans without writing code.
- **Agriculture:** Scientists can rapidly train models to detect plant diseases or pests using drone imagery.
- **Ecology:** Environmental researchers segment land cover or species distributions from satellite data using a simple interface.

# Acknowledgements

.........................

# References

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. GitHub repository. https://github.com/ultralytics/ultralytics
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 779–788.

