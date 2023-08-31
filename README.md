# Taxonomy Adaptive Cross-Domain Adaptation in Medical Imaging via Optimization Trajectory Distillation - ICCV 2023
## Introduction and Installation
In this project, we propose optimization trajectory distillation to present a unified framework for addressing the two technical challenges in taxonomy-adaptive domain adaptation.
Pytorch 1.12.1 with CUDA 11.3 is adopted.

Additionally, please install the tiatoolbox package, following the instructions given in https://github.com/TissueImageAnalytics/tiatoolbox.
## Data
### Data Introduction
On the nuclei segmentation and recognition benchmark, we use subsets of PanNuke and Lizard as the source and target dataset, respectively.
Please refer to [PanNuke](https://arxiv.org/abs/2003.10778) and [Lizard](https://arxiv.org/abs/2108.11195) for data access.

**If you are using these datasets in your research, please also remember to cite their original work.**
### Data Preparation
Please follow the detailed instructions in code to type in the correct path for training and test set.

## Model Training and Evaluation
```
ulimit -n 65000
source activate [your_environment]
python run.py
cd ./infer/
python run_script.py
```

## Contact
For any questions, please contact jiananfan0604@outlook.com.
