# Sentiment Analysis of IMDb Movie Reviews Using Logistic Regression and Fine-Tuned BERT

## Project Overview

This project compares a traditional machine learning model with a transformer-based Large Language Model (LLM) for binary sentiment classification of IMDB movie reviews.

Two approaches were implemented:

- Logistic Regression with TF-IDF features (Baseline)
- Fine-Tuned BERT (`bert-base-uncased`)

The objective is to evaluate whether fine-tuning a pre-trained BERT model improves sentiment classification performance compared with a traditional machine learning approach.

---

## Dataset

**Dataset:** IMDB Movie Reviews

- Source: Hugging Face IMDb Dataset
- Total dataset size: 50,000 labelled movie reviews
- Binary classification:
  - Positive
  - Negative

A subset of the dataset was used for faster experimentation:

| Split | Reviews |
|--------|---------:|
| Training | 4,000 |
| Validation | 1,000 |
| Testing | 2,000 |

---

## Project Workflow

1. Load IMDb dataset
2. Split into training, validation, and testing sets
3. Train TF-IDF + Logistic Regression baseline
4. Tokenise text using the BERT tokenizer
5. Fine-tune the pre-trained `bert-base-uncased` model
6. Evaluate both models
7. Compare model performance

---

## Technologies Used

- Python
- Google Colab
- Hugging Face Transformers
- Hugging Face Datasets
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

## Models

### Logistic Regression

- TF-IDF feature extraction
- Binary sentiment classification
- Traditional machine learning baseline

### Fine-Tuned BERT

- Pre-trained `bert-base-uncased`
- Fine-tuned using the IMDb training dataset
- Binary sentiment classification

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

---

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|--------|---------:|----------:|-------:|---------:|
| Logistic Regression | 85.05% | 83.60% | 87.20% | 85.36% |
| Fine-Tuned BERT | **90.95%** | **90.58%** | **91.40%** | **90.99%** |

The fine-tuned BERT model achieved the highest performance across all evaluation metrics, demonstrating its ability to better understand contextual information in movie reviews.

---

## Repository Structure

```
├── Assignment_3_Individual_project_Large_language_models.ipynb
├── report.pdf
├── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/deepakraj-04/Individual-Project-Large-Language-Models.git
```

Install the required packages:

```bash
pip install transformers
pip install datasets
pip install torch
pip install scikit-learn
pip install pandas
pip install matplotlib
```

or

```bash
pip install -r requirements.txt
```

---

## Running the Project

Open the notebook in:

- Google Colab (Recommended)

or

- Jupyter Notebook

Run all cells from top to bottom.

---

## Future Improvements

Possible future enhancements include:

- Training on the complete IMDb dataset
- Hyperparameter tuning
- Comparing additional transformer models such as RoBERTa and DistilBERT
- Testing on other sentiment analysis datasets

---

## Author

**Deepak Raj**

MSc Data Science

University Assignment – Large Language Models (LLMs)

---

## License

This repository is created for educational purposes as part of an MSc coursework project.
