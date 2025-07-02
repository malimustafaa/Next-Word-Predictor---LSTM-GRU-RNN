# Next Word Prediction with LSTM (Hamlet - NLTK)

This project builds an LSTM-based model to predict the next word in a sequence, trained on *Hamlet* by William Shakespeare from the NLTK Gutenberg corpus.

## 📘 Dataset
- Source: `nltk.corpus.gutenberg`
- Text: `shakespeare-hamlet.txt`

## 🧠 Model Overview
- **Embedding Layer**: Converts tokens to dense vectors.
- **LSTM Layers**: Two stacked LSTM layers (150 + 100 units).
- **Dropout**: Regularization to reduce overfitting.
- **Dense Layer**: Softmax over vocabulary for next-word prediction.

## ⚙️ Training
- Loss: `categorical_crossentropy`
- Optimizer: `Adam`
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`

## 🔍 Example
```python
predict_next_word(model, tokenizer, "to be or", max_sequence_len)
# Output: 'not'

💾 Notes
Tokenizer saved using pickle.

Vocabulary size controlled via Keras Tokenizer.

Validation metrics monitored to avoid overfitting.

📌 Requirements
TensorFlow

NLTK

NumPy
