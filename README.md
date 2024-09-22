# Trigram Language Model for Essay Classification

This project implements a trigram language model in Python to classify essays based on their perplexity. It works by training models on high and low proficiency essays and computing perplexity scores to predict the skill level of unseen essays.

## Key Features
- **N-gram Extraction**: Handles unigram, bigram, and trigram extraction with padding.
- **Smoothed Probabilities**: Uses linear interpolation to smooth probabilities.
- **Perplexity Calculation**: Measures how well the model predicts the test data.
- **Essay Classification**: Classifies essays by comparing perplexity scores.

## How to Run
1. Clone the repository and install any required dependencies (if needed).
2. Run the `trigram_model.py` script:
   ```bash
   python trigram_model.py
