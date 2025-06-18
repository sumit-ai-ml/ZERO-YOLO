---
title: 'ZERO-YOLO: A No-Code GUI for Training Custom Dataset on YOLO Segmentation Models'
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
    orcid: 0000-0003-3791-2620
    affiliation: 1
  - name: Satyasaran Changdar
    orcid: 0000-0002-7704-8315
    affiliation: 2
  - name: Erik Bjørnager Dam
    orcid: 0000-0002-8888-2524
    affiliation: 1
affiliations:
  - name: Department of Computer Science, University of Copenhagen
    index: 1
  - name: Department of Food Science, University of Copenhagen
    index: 2
date: 18 June 2025
bibliography: paper.bib
---


# Summary

YOLO (YOLOv11/YOLOv8) segmentation models routinely achieve impressive scores on benchmark datasets (i.e. COCO dataset [@lin2014coco]), yet deploying them on a *new* dataset still demands (i) command-line / coding proficiency, and (ii) careful data re-formatting that many domain scientists lack.  Commercial “auto-ML” platforms (e.g. Roboflow)  [@roboflow] simplify the process but require cloud upload of your own (possibly sensitive) image dataset and frequently impose paywalls or restrictive licences.

**ZERO-YOLO** closes this usability gap. It bundles the complete YOLO segmentation training pipeline into a single, Dockerised web application built with *Streamlit*. Researchers from biology, medicine, materials science, or any other field can—in a few clicks—convert local image–mask pairs into YOLO format, fine-tune a pretrained model, monitor training metrics live, and visualise predictions, all while their data never leave the workstation. 


# Statement of Need

Image segmentation is the process of dividing an image into distinct regions that correspond to different objects, structures, or classes of interest. It is a fundamental task in computer vision and image analysis, with critical applications across a wide range of fields.:

- **In healthcare**, segmentation helps radiologists delineate tumors, organs, or lesions in medical scans.
- **In agriculture**, researchers use segmentation to analyze crop health or identify pests from aerial imagery.
- **In environmental science**, ecologists segment satellite or drone images to study land use, habitats, or pollution.
- **In manufacturing**, engineers use segmentation for quality control and defect detection.
- **In education and research**, segmentation is vital for teaching and developing new computer vision techniques.

A YOLO segmentation model extends the “You Only Look Once” detector to predict pixel-precise masks and class labels in a single, end-to-end pass. Its lightning-fast inference and solid accuracy make it a go-to choice for real-time, resource-efficient vision tasks. While it shows amazing performance on benchmark datasets, *domain specialists* still face two practical barriers:

1. **Workflow complexity.** Preparing config files, writing training scripts, and troubleshooting CUDA/PyTorch errors deter non-ML experts [@ultralytics-docs].  
2. **Data sovereignty.** Institutions handling medical or proprietary imagery cannot upload data to third-party services [@roboflow].

ZERO-YOLO addresses both. It eliminates coding entirely and keeps the full workflow on-premise under a permissive MIT licence, empowering scientists to run high-quality object-detection experiments without cloud dependencies or legal entanglements. The installation and usage instructions (both written and video) for the software can be found at: https://github.com/sumit-ai-ml/ZERO-YOLO

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
 
# Software Description

| Aspect                | Detail |
|-----------------------|--------|
| **Language / stack**  | Python 3.10; Streamlit UI; PyTorch backend; Ultralytics YOLO engine |
| **Distribution**      | One-command Docker Compose (CPU or NVIDIA GPU profile) |
| **Input**             | Image folder + segmentation-mask folder + *label\_names.xlsx* listing classes |
| **Output**            | Trained `*.pt` weights, live training plots, interactive mask overlays |
| **Licence**           | MIT |

Internally the app launches a YOLO training process with user-selected epochs and model size, streams logs to the UI, and writes artefacts to the `runs/` directory. A small custom wrapper auto-generates YOLO YAML configs and converts segmentation masks to COCO-style polygons when needed.


# Domain Impact

ZERO-YOLO democratizes access to advanced segmentation, benefiting a wide spectrum of domains:

- **Healthcare:** Medical image segmentation (e.g., CT, MRI) without coding.
- **Agriculture:** Crop monitoring, pest detection from field/drone imagery.
- **Environmental Science:** Segmentation of satellite/drone imagery for ecological studies.
- **Education:** Enables hands-on teaching of deep-learning segmentation without programming barriers.


# References

