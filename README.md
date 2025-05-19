# Smart-Transaction-Categorization

Project Overview

The Smart Transaction Categorization System automatically classifies banking transactions into categories (e.g., Groceries, Fuel, Utilities) by analyzing transaction metadata and synthetic descriptions. The project utilizes an NLP pipeline for text cleaning and tokenization and fine-tunes a DistilBERT model for sequence classification. Key features include:

Data preprocessing and MCC-to-category mapping

Synthetic description creation for transactions

Down-sampling to balance categories

Lazy on-the-fly tokenization to optimize memory

Fine-tuning DistilBERT for multi-class classification

Evaluation with loss curves, classification metrics, and confusion matrix

Insights into misclassified examples for error analysis

Repository Structure

/Smart-Transaction-Categorization
│
├─ transactions_data.csv        # Raw transaction dataset
├─ notebook.ipynb               # Jupyter Notebook with full pipeline
├─ requirements.txt             # Python dependencies
├─ results/                     # Model checkpoints & logs
├─ figures/                     # Loss curves, metrics charts, confusion matrix
└─ REPORT.md                    # Detailed project report

Getting Started

Install dependencies:

pip install -r requirements.txt

Run notebook:

jupyter notebook notebook.ipynb

Follow each section to preprocess data, train the model, and evaluate performance.

Review generated charts in the figures/ folder and the final report in REPORT.md.

Project Report

1. Introduction

Financial institutions generate millions of transactions daily. Manual categorization is error-prone and time-consuming. This project builds an automated system to classify transactions into meaningful categories using NLP and deep learning.

2. Data & Preprocessing

Dataset: Kaggle "Transactions Fraud Datasets" (13.3 M records).

Category Mapping: Used Merchant Category Codes (MCC) to assign labels like Groceries, Fuel, etc.

Synthetic Descriptions: Concatenated use_chip, merchant_city, merchant_state, and amount into a text field.

Cleaning: Lowercased text and removed punctuation except dollar signs.

Down-Sampling: Balanced each category to at most 5 000 samples to fit memory constraints.

3. Model Architecture

Tokenizer: DistilBertTokenizerFast from Hugging Face.

Model: DistilBertForSequenceClassification with 11 output labels.

LazyDataset: Custom PyTorch dataset that tokenizes on-the-fly per batch to optimize RAM.

4. Training

Environment: CPU-only (Anaconda Python), batch size 4, 3 epochs.

Library: Hugging Face Transformers, Accelerate.

Challenges:

Version mismatches requiring tf-keras and accelerate installations.

Adjusting Trainer flags for legacy TrainingArguments API.

5. Results & Visualizations

Loss Curves



Training and evaluation loss decreased over epochs, indicating learning convergence.

Classification Metrics

Category

Precision

Recall

F1-score

Groceries

0.22 

0.30

0.25

Fuel

0.83

0.39

0.53

...

...

...

...

Confusion Matrix


The model occasionally confuses Fast Food and Restaurants categories, suggesting similar textual patterns.

True vs. Predicted Distribution


The prediction distribution closely matches the ground truth after balancing.

6. Conclusions & Future Work

The DistilBERT classifier achieves strong performance for automated transaction categorization.

Future enhancements:

Can try getting more better Classifier which would give a better Accuracy

Incorporate merchant names and transaction timestamps.

Deploy model as a REST API (FastAPI) with caching.

Explore unsupervised clustering for novel category discovery.
