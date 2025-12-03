# Applied Deep Learning – Assignment 1 (Initiate)

**Working Title:** Federated-Continual Learning with an Adaptive Policy Controller (LLM-ready roadmap)  
**Project Type:** Reproduction + Extension (research-engineering hybrid)  
**Student:** Somayeh Shami  
**Course:** Applied Deep Learning (PhD, TU Graz — co-registered at TU Wien)  

---

## 1. Context and Motivation

In many real-world applications, data are **distributed across multiple devices** (federated setting) and the **data distribution changes over time** (continual setting).  
- **Federated Learning (FL)** trains a global model on several clients without moving the raw data off the devices.  
- **Continual Learning (CL)** trains a model on a sequence of tasks or data batches over time.

**Federated-Continual Learning (FCL)** combines both: multiple clients each receive a stream of data over time. A key challenge in CL and FCL is **catastrophic forgetting** – when the model updates on new data and loses performance on previously seen classes or tasks.

In this project, I work on a standard **image classification** task on **CIFAR-100** using a **ResNet-18** backbone. The goal is to train a federated-continual classifier that maintains good overall accuracy across all classes while reducing catastrophic forgetting by using an **adaptive policy controller** that tunes learning rate and replay ratio during training.

---

## 2. Topic & Goal
This project reproduces the baseline setup from my previous work on **Federated-Continual Learning (FCL)** using CIFAR-100 and a ResNet-18 backbone, then extends it with an **adaptive policy controller** that dynamically tunes hyperparameters (learning rate, replay ratio) during training.  
The goal is to improve generalization and reduce catastrophic forgetting.  
This controller design also lays the foundation for a future **LLM-guided policy module** as part of my ongoing PhD research.

---

## 3. Related Work
1. **Shami, S. et al. (2024)** – *Federated Continual Learning for Vision-Based Plastic Classification in Recycling*, *Waste Management Journal (Elsevier)*.  
2. **Aberger, J., Shami, S. et al. (2024)** – *Prototype of AI-Powered Assistance System for Digitalisation of Manual Sorting*, *Waste Management Journal (Elsevier)*.  
3. **Shami, S. et al. (2024)** – *Comparative Analysis of Transfer and Continual Learning for Particle Classification in Plastic Sorting*, *Recy&DepoTech Conference 2024*, pp. 585–592.  
4. **Shami, S. et al. (2024)** – *A Vision-Based Trash Particle Classification System for Sorting Facilities*, *13th Science Congress on Circular and Resource Economy*, pp. 35–42.

---

## 4. Approach Summary
### Baseline (Reproduction)
- **Dataset:** CIFAR-100 (50 k train / 10 k test).  
- **Split:** 45 k train / 5 k val / 10 k test.  
- **Federated setup:** 4 clients, equal-size IID splits of the 45 k training portion.  
- **Continual setup:** 7 CL batches per client (≈46 % initial + 6 increments).  
- **Model:** ResNet-18 (ImageNet-pretrained).  
- **Training:** Adam (lr = 1e-4), batch = 256, replay ratio = 0.5, num_workers = 4.  
- **Validation:** Early stopping (patience = 5) to curb overfitting.  
- **Logging:** `fcl_run_results.csv`, `fcl_run_summary.csv`, `fcl_run_cl_batches.csv`.

### Extension (Controller)
- Add a **policy controller** to adapt `lr` and `replay ratio` each FL round using validation signals (accuracy trend + forgetting proxy).  
- Evaluate improvements over the baseline in **accuracy** and **forgetting metrics**.

### PhD Roadmap
Future extension will replace the rule-based controller with an **LLM-guided policy**, prompted by run summaries and guided by safety / reproducibility constraints.

---

## 5. Dataset Description
- **Dataset:** CIFAR-100 – 100 classes, 32×32 RGB images.  
- **Transforms:** Resize to 224, normalize to ImageNet stats, random horizontal flip.  
- **Federated split:** Equal IID across 4 clients (45 k train).  
- **Continual schedule:** 7 batches per client (1 large initial + 6 increments).

---

## 6. Work Breakdown & Time Estimates

| Work Package | Tasks | Time (h) |
|---------------|--------|----------:|
| **Dataset Preparation** | Download, verify splits, build 4-client and 7-batch splits | 6 |
| **Network Design & Build** | Implement ResNet-18 + replay + early stopping + logging | 8 |
| **Training & Fine-Tuning** | Baseline + controller runs, evaluation & plots | 20 |
| **Demo / Presentation Artifact** | Create notebook, plots, README polish | 6 |
| **Final Report** | Baseline vs controller analysis, discussion | 10 |
| **Presentation Slides** | Summarize problem, method, results, future plan | 6 |
| **Total** | | **≈ 56 hours** |

---

## 7. Current Status
- ✅ Baseline FCL training + validation implemented.  
- ✅ Replay buffer and 7-stage CL schedule in place.  
- ✅ Results exported to CSV for reproducible comparison.  
- ✅ Initial controller integration tested (promising gains).