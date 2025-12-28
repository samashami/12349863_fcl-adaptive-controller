# Applied Deep Learning – Assignments 1–3

**Working Title:** Federated-Continual Learning with an Adaptive Policy Controller (LLM-ready roadmap)  
**Project Type:** Reproduction + Extension (research-engineering hybrid)  
**Student:** Somayeh Shami  
**Course:** Applied Deep Learning (PhD, TU Graz — co-registered at TU Wien)  

---

# Assignment 1 – Initiate

## 1. Context and Motivation
In many real-world applications, data are **distributed across multiple devices** (federated setting) and the **data distribution changes over time** (continual setting).

- **Federated Learning (FL)** trains a global model across several clients without sharing raw data.  
- **Continual Learning (CL)** trains a model on a sequence of evolving tasks or data batches.  

**Federated-Continual Learning (FCL)** combines both: each client receives a data stream over time.  
A major challenge is **catastrophic forgetting**—when new updates overwrite previously learned knowledge.

This project builds an **image classification** pipeline on **CIFAR-100** using a **ResNet-18** backbone and evaluates how an **adaptive policy controller** can reduce forgetting and improve generalization.

---

## 2. Topic & Goal
Reproduce the baseline FCL pipeline from previous work (CIFAR-100 + ResNet-18 + replay).  
Then extend it by integrating a **dynamic hyperparameter controller** that adjusts learning rate and replay ratio during training.

This serves as both a course project and as groundwork for a future **LLM-guided policy controller** in my PhD research.

---

## 3. Related Work
1. Shami, S. et al. (2025) — *Federated Continual Learning for Vision-Based Plastic Classification*.  
2. Aberger, J., Shami, S. et al. (2025) — *AI-Powered Assistance System for Manual Sorting*.  
3. Shami, S. et al. (2024) — *Comparative Analysis of Transfer and Continual Learning*.  
4. Shami, S. et al. (2024) — *Vision-Based Trash Particle Classification System*.

---

## 4. Approach Summary

### Baseline (Reproduction)
- Dataset: CIFAR-100 (50k train, 10k test)
- Split: 45k train / 5k val / 10k test  
- Federated: 4 IID clients  
- Continual: 5 data batches per client  
- Model: ResNet-18 (ImageNet-pretrained)  
- Training: Adam (lr=1e-4), batch size=128, replay ratio=0.5  
- Logging: CSV files for reproducibility

### Extension (Controller)
Rule-based controller adjusting:
- learning rate  
- replay ratio  
based on validation accuracy trends and forgetting proxy.

### Long-Term (PhD)
LLM-guided policy controller using run metadata.

---

## 5. Dataset Description
- CIFAR-100 (100 classes, 32×32 RGB)  
- Preprocessing: resize to 224, ImageNet normalization  
- Federated: equal IID split across 4 clients  
- Continual: 5-stage batch schedule

---

## 6. Work Breakdown & Time Estimates (Assignment 1)

| Work Package | Tasks | Time (h) |
|---------------|--------|----------:|
| Dataset Preparation | Build splits, 4 clients, 5 batches | 6 |
| Network Build | ResNet-18, replay, early stopping, logging | 8 |
| Training & Fine-Tuning | Baseline + controller runs | 20 |
| Demo & Figures | Notebook, plots, README polish | 6 |
| Final Report | Analysis & discussion | 10 |
| Presentation Slides | Method + results | 6 |
| **Total** | | **≈ 56 h** |

---

# Assignment 2 – Hacking

## 7. Error Metric and Targets

### Primary Metric  
**Top-1 Test Accuracy** on CIFAR-100.

### Secondary Metric (research-oriented)  
Average accuracy across CL batches + simple forgetting measure.

### Targets  
- Baseline CL: **≥ 60%** top-1 accuracy  
- Tuned/controller: **≥ 65%** top-1 accuracy + reduced forgetting

## 8. Baseline Pipeline & Results (Assignment 2)

### 8.1 Experimental Setup

For Assignment 2, a simplified continual-learning-only baseline (single-client simulation) is implemented as a first step toward the full Federated-Continual Learning setup outlined in Assignment 1. All substantial experiments for Assignment 2 were executed on **Kaggle (GPU runtime)**.

- Model: ResNet-18 (ImageNet-pretrained)
- Dataset: CIFAR-100
- Train / validation / test split: 45k / 5k / 10k
- Continual setup: 5 continual-learning batches (single client simulation)
- Optimizer: Adam (lr = 1e-4)
- Batch size: 128
- Epochs per CL batch: 2

### 8.2 Achieved Results

Baseline experiment command:

```bash
python src/train_baseline.py \
    --epochs-per-batch=2 \
    --num-cl-batches=5 \
    --batch-size=128
```

Final results (last continual batch):

- **Top-1 test accuracy:** **61.39 %**
- **Validation accuracy:** **61.76 %**
- Results file: `results/runs/20251209_121214/baseline_results.csv`

The baseline model meets the predefined target of **≥ 60 % top-1 test accuracy**, demonstrating a functional end-to-end continual image classification pipeline. Further improvements via adaptive controllers or hyperparameter tuning are left as future work.

## 9. Time Tracking (Assignment 2)

| Work Package                   | Planned (h) | Spent (h) |
|--------------------------------|------------:|----------:|
| Dataset preparation & splits   | 6           | 3 |
| Baseline implementation        | 8           | 7 |
| Training & hyperparam search   | 20          | 9 |
| Controller integration         | 8           | 0 |
| Analysis & plots               | 6           | 3 |
| README / documentation         | 4           | 3 |
| **Total**                      | **52**      | **25** |


## 10. Repository Structure (current & planned)

```text
12349863_fcl-adaptive-controller/
├── README.md
├── SUBMISSION.md
├── requirements.txt              # planned
│
├── src/                           # planned
│   ├── data.py
│   ├── model.py
│   ├── train_baseline.py
│   ├── controller.py
│   └── __init__.py
│
├── notebooks/                     # planned
│   └── exploration.ipynb
│
└── results/                       # current & planned
    └── runs/
        ├── 20251203_164301/
        │   └── baseline_results.csv
        └── 20251209_121214/
            └── baseline_results.csv
```

# Assignment 3 – Demo Application

This assignment provides a minimal, reproducible demo of a Federated Continual Learning (FCL) system.

The goal is to demonstrate:
- end-to-end training
- model checkpointing
- inference using a trained model
- reproducibility via logged CSV outputs

## Demo Overview

The demo consists of:
- A lightweight FCL training script
- A saved global model checkpoint
- A simple inference script
- Logged metrics for inspection

### Run Training
```
python -m assignment3.src.run_fcl --rounds 1 --epochs 1 --device cpu
```
