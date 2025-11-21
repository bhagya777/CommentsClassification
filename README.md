# 🤖 Contextual Sentiment & Intent Classification using XLM-RoBERTa

## 🌟 Project Summary

This project implements a state-of-the-art **Transformer-based model (XLM-RoBERTa)** to classify unstructured textual comments into one of four primary intent categories: **Complaint, Demands, Praise, or Questions**.

The goal is to build a robust system for a financial or large-scale customer service environment, where **real-time risk assessment** and priority routing of customer feedback are critical business requirements.

The solution achieved strong performance, demonstrating an accuracy and F1 score of approximately **89%** and a weighted ROC AUC of **0.97** on the validation set.

---

## 🛠️ Technical Stack & Architecture

| Component | Technology / Skill Highlight | Purpose in Project |
| :--- | :--- | :--- |
| **Model Architecture** | **XLM-RoBERTa-base (Transformer)** | Fine-tuning a powerful, pre-trained, multilingual model for superior contextual understanding of mixed-language text data. |
| **Frameworks** | **Hugging Face `transformers` & PyTorch** | Utilized the `Trainer` API for efficient, optimized training and rigorous evaluation on GPU resources. |
| **Data Augmentation**| **Synonym Replacement (NLTK)** | Implemented a technique to effectively **double the training dataset size**, mitigating overfitting and enhancing model generalization. |
| **Optimization** | **Gradient Accumulation** | Applied gradient accumulation steps (`gradient_accumulation_steps=2`) to increase the **effective batch size** and maintain training stability on limited GPU memory. |
| **Evaluation** | **ROC AUC (Multi-Class OVR)** | Used ROC AUC as the primary metric, providing a robust measure of model performance across all four classification categories, especially in scenarios with class imbalance. |

---

## 🚀 Key Implementation Steps

### 1. Data Preparation and Augmentation
* **Label Creation:** Consolidated the four initial label columns (`complaint`, `demands`, `praise`, `questions`) into a single **multi-class label** using `np.argmax` to define the dominant intent.
* **Data Augmentation:** The custom `synonym_replacement` function (using NLTK's `wordnet`) was applied, effectively doubling the dataset size to 8,000 samples.

### 2. Model Fine-Tuning
* **Tokenizer:** The `xlm-roberta-base` tokenizer prepared inputs with truncation and padding to `max_length=128`.
* **Custom Dataset:** A PyTorch `CustomDataset` class was implemented for efficient data handling.
* **Training Configuration:** The model was fine-tuned for **5 epochs** with a learning rate of $2e-5$. The model was configured to load the best version based on the validation **ROC AUC** score.

### 3. Prediction Pipeline
* The final model was used to predict probabilities for the test set.
* **Softmax** was explicitly applied to the logits to ensure the final output for each comment consisted of four valid probability scores that sum to 1, ready for submission or downstream consumption.

---

## 📊 Performance Summary

The model's final performance after fine-tuning demonstrates excellent classification capacity for the task:

| Metric | Score |
| :--- | :--- |
| **Evaluation Loss** | 0.368 |
| **ROC AUC (Weighted)** | **0.973** |
| **F1 Score (Weighted)** | **0.890** |
| **Accuracy** | **0.890** |

---

## ⚙️ How to Run This Project

### Prerequisites
* Python 3.8+
* GPU access (Recommended for Transformer models)
* The original `train.csv` and `test.csv` datasets.

### Installation

```bash
# Install core data science and deep learning libraries
pip install numpy pandas scikit-learn torch

# Install Hugging Face Transformers and the Trainer API
pip install transformers

# Install NLTK (needed for the synonym replacement function)
pip install nltk