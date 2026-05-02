#  Structured Model Pruning & Network Surgery: Sanskrit MNIST

This repository demonstrates **Structured Model Pruning** (Network Surgery) applied to a custom ResNet architecture using TensorFlow/Keras. 

Instead of merely masking weights to zero (which theoretically reduces model size but doesn't decrease actual RAM/VRAM usage or inference time), this project performs **physical extraction of entire residual blocks**. By transferring the surviving weights to a physically smaller architecture and applying post-surgery fine-tuning, the model size is cut by **52.5%** without losing a single drop of accuracy.

##  The Results

| Metric | Original Model | Post-Surgery (Shock) | Fine-Tuned (Final) |
| :--- | :---: | :---: | :---: |
| **Parameters** | 704,990 | 334,814 | **334,814** |
| **Model Size Reduction**| - | - 52.5% | **- 52.5%** |
| **Test Accuracy** | 99.50% | 5.49% | **99.50%** |

*Note: After surgically removing 2 massive blocks, the model experiences an initial "shock," dropping to 5.49% accuracy. However, after only **5 epochs** of fine-tuning with a very low learning rate (1e-4), the remaining layers completely recover the lost knowledge, matching the original 99.50% accuracy!*

##  Dataset: Sanskrit MNIST
We used the highly challenging **Sanskrit MNIST** dataset, which consists of 62 classes of handwritten Devanagari characters. It provides a much more complex and culturally rich alternative to the traditional Latin digit MNIST.

*   **Classes:** 62 (Vowels, Consonants, Numerals)
*   **Image Size:** 32x32, Grayscale
*   **Total Images:** 31,000

> **Note:** The dataset is excluded from this repository via `.gitignore` to keep the repo clean. 
> You can download the dataset here: **[Sanskrit MNIST on Kaggle](https://www.kaggle.com/datasets/maulikgajera/sanskrit-mnist)**.
> Extract it into a folder named `Sanskrit_MNIST` in the root directory before running the scripts.

##  Project Architecture & Methodology

The project is heavily modularized to separate data pipeline, model building, training, and pruning logic.

### 1. Phase 1: Base Training (`1_train.py`)
Trains a custom, modular ResNet-style architecture from scratch on the Sanskrit dataset. It uses `ReduceLROnPlateau` and `ModelCheckpoint` callbacks to achieve the best possible baseline convergence.
*   **Result:** 99.50% Test Accuracy.

### 2. Phase 2: Weight Analysis & Physical Pruning (`2_analyze_and_prune.py`)
This is the core of the project. It evaluates the `L2 Norm`, `Variance`, and `Dead Weight Ratio (<1e-4)` of every `Conv2D` layer in the trained model.
*   Instead of Unstructured Pruning (setting weights to 0), we identify the "laziest" blocks (lowest variance) and completely omit them from a new, smaller model definition.
*   **Network Surgery:** We use a custom `transfer_weights()` function to copy the learned weights from the surviving layers of the massive model and inject them into the new, physically smaller architecture.

### 3. Phase 3: Shock Recovery & Fine-Tuning (`3_finetune.py`)
Removing layers physically disconnects established data flows, sending the model into a "shock" state (accuracy drops to 5.49%).
*   We compile the pruned model with a very low Learning Rate (`1e-4`) to prevent destroying the transferred weights.
*   After just 5 epochs of fine-tuning, the surviving layers adapt to the missing blocks and restore the model to its original **99.50%** accuracy, proving the principles behind the *Lottery Ticket Hypothesis* in a practical, production-ready manner.

##  How to Run

1. Clone the repository.
2. Download the dataset from Kaggle and place the images in a folder named `Sanskrit_MNIST`.
3. Install dependencies: `pip install tensorflow matplotlib numpy`
4. Run the pipeline in order:
   ```bash
   python 1_train.py
   python 2_analyze_and_prune.py
   python 3_finetune.py