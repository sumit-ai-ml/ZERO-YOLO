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
    affiliation: 2
  - name: Erik Bjørnager Dam
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Department of Computer Science, University of Copenhagen
    index: 1
  - name: Department of Food Science, University of Copenhagen
    index: 2
date: 2025-06-03
repository: https://github.com/sumit-ai-ml/ZERO-YOLO

---


# Summary

Image segmentation is central to diverse domains, from healthcare to agriculture, yet remains out of reach for many due to the programming expertise required. **ZERO-YOLO** addresses this gap by offering a truly no-code graphical user interface (GUI) for the entire workflow of training and evaluating YOLO-based segmentation models. Built on Streamlit, ZERO-YOLO empowers researchers, educators, and domain experts—regardless of coding background—to harness the power of modern deep learning for segmentation tasks.

# Statement of Need

Image segmentation is the process of dividing an image into distinct regions that correspond to different objects, structures, or classes of interest. It is a fundamental task in computer vision and has critical applications across a wide range of fields:

- **In healthcare**, segmentation helps radiologists delineate tumors, organs, or lesions in medical scans.
- **In agriculture**, researchers use segmentation to analyze crop health or identify pests from aerial imagery.
- **In environmental science**, ecologists segment satellite or drone images to study land use, habitats, or pollution.
- **In manufacturing**, engineers use segmentation for quality control and defect detection.
- **In education and research**, segmentation is vital for teaching and developing new computer vision techniques.

Despite its broad relevance, modern segmentation solutions such as YOLO require specialized coding skills, familiarity with command-line tools, and careful data preparation. This creates a barrier for many domain experts—including clinicians, biologists, teachers, and industrial researchers—who need segmentation tools but do not have programming backgrounds.

**ZERO-YOLO** addresses this gap by providing a truly no-code, graphical user interface for the entire YOLO segmentation workflow. Users can prepare data, convert annotations, configure and train models, and visualize results—all without writing code. This lowers the entry barrier, enabling experts from any discipline to apply cutting-edge deep learning in their work.

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
  - Direct integration with Ultralytics YOLOv8 or any other advanced version of YOLO.
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



# References

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. GitHub repository. https://github.com/ultralytics/ultralytics
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 779–788.

