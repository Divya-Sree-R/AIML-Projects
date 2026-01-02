# Sentiment Analysis using TF-IDF and Logistic Regression

## Overview

This project demonstrates a basic Natural Language Processing (NLP) pipeline for sentiment analysis using classical machine learning techniques. The objective is to classify text reviews as Positive or Negative based on their content.

The implementation follows a clean, notebook-oriented, academic approach without deployment, file handling, or external dependencies beyond standard ML libraries.

---

## Problem Statement

Given a short text sentence or review, predict whether the sentiment expressed is:

- Positive (1)
- Negative (0)

This is a binary text classification problem.

---

## Dataset Description

A small, manually created dataset is used for demonstration purposes.

- Text: Short user opinions and reviews
- Label:
  - 1 → Positive sentiment
  - 0 → Negative sentiment

Example entries:

| Text                        | Label |
|----------------------------|-------|
| I love this product        | 1     |
| Worst experience ever      | 0     |
| Excellent quality          | 1     |
| Completely disappointing   | 0     |

The dataset size is intentionally small to focus on pipeline clarity rather than model performance.

---

## Methodology

The project follows a standard NLP workflow:

### 1. Text Vectorization
- TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text into numerical features.
- Stop words are removed to reduce noise.

### 2. Model Selection
- Logistic Regression is used as the classifier.
- Chosen for its simplicity, interpretability, and suitability for binary classification.

### 3. Training and Testing
- The dataset is split into training and testing sets.
- The model is trained on the training data and evaluated on unseen test data.

---

## Evaluation Metrics

The following metrics are used to evaluate the model:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics provide a clear understanding of classification performance.

---

## Sample Prediction

The trained model can predict sentiment for new, unseen sentences.

Example:

Input:
The product is very good

Output:
Positive

This demonstrates successful inference using the trained NLP pipeline.

---

## Tools and Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn

---

## Key Highlights

- Clean and minimal NLP pipeline
- No file input/output operations
- No deployment or packaging logic
- Fully notebook-based implementation
- Easy to explain and academically sound

---

## Conclusion

This project provides a clear and correct implementation of sentiment analysis using traditional NLP techniques. It serves as a strong foundation for understanding text preprocessing, feature extraction, and machine learning-based classification in NLP.

