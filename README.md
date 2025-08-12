# :sauropod: Grounding DINO baseline for MARS2_Track1
This repository provides **[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)**-baseline for Multimodal Reasoning Competition Track1 [(VG-RS)](https://eval.ai/web/challenges/challenge-page/2552/overview).

# Set-Up
## Environment
```bash
conda create -n grounddino
conda activate grounddino
conda install python==3.9.18
pip install groundingdino-py
```
## Model
```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

## Directory Structure
```
project_root/
├── inference.py                               # Main script
├── config.py                                  # Model configuration file
├── weights                                    # Pre-trained weights
├── images/                                    # Folder with images
│   └── *.jpg / *.png
├── VG-RS-question.json                        # Input questions and image paths
└── predict_grounding_full_3b.json             # Output predictions (bounding boxes)
```

# Inference
```bash
python inference.py
```
To save the results with bounding boxes,
```bash
python inference.py --save_results
```