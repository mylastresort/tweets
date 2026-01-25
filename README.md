# Tweet Sentiment Analysis

A Python-based toolkit for analyzing and classifying tweet sentiments using Natural Language Processing (NLP) and Machine Learning techniques. This project provides tools for text cleaning, tokenization, vectorization, and sentiment classification of tweets into positive, negative, or neutral categories.

## Quick Access — Interactive Notebooks

For the fastest way to explore results and run the complete analysis, open the Jupyter notebooks:

| Notebook | Description |
|----|----|
| **[tweets.ipynb](tweets.ipynb)** | Main analysis notebook — complete pipeline with outputs |
| [tweets_analysis.ipynb](tweets_analysis.ipynb) | Additional exploratory analysis |
| [tweets_similarity.ipynb](tweets_similarity.ipynb) | Tweet similarity comparisons |

```bash
jupyter notebook tweets.ipynb
```

## Features

* **Text Preprocessing**: Clean tweets by removing mentions, URLs, hashtags, and punctuation
* **Multiple Tokenization Methods**: Support for various stemming (Porter, Snowball, Lancaster) and lemmatization approaches
* **Spell Correction**: Automatic misspelling correction using SymSpell
* **Vectorization Options**: Count Vectorizer, Binary Vectorizer, and TF-IDF Vectorizer
* **Machine Learning Classification**: Train sentiment classifiers using Random Forest
* **Interactive Notebooks**: Jupyter notebooks for exploratory analysis and experimentation

## Data

The project includes pre-processed tweet datasets:

* `./data/processedNegative.csv` - Tweets labeled as negative sentiment
* `./data/processedPositive.csv` - Tweets labeled as positive sentiment
* `./data/processedNeutral.csv` - Tweets labeled as neutral sentiment

## Module Reference

| Module | Description |
|----|----|
| `config.py` | Load and merge tweet datasets into a unified DataFrame |
| `cleaning.py` | Text preprocessing (remove mentions, URLs, hashtags, punctuation) |
| `tokenizer.py` | Tokenization with stemming, lemmatization, and spell correction |
| `vectorizer.py` | Convert text to numerical features (BoW, Binary, TF-IDF) |
| `train.py` | Train and evaluate machine learning classifiers |

## Getting Started

### Requirements

#### Software Requirements

* Python 3.8 or higher
* pip (Python package manager)

#### Python Dependencies

* pandas
* scikit-learn
* nltk
* symspellpy

### Installation


1. **Clone the repository**

   ```bash
   git clone https://github.com/mylastresort/tweets
   cd tweets
   ```
2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # or
   venv\Scripts\activate     # On Windows
   ```
3. **Install dependencies**

   ```bash
   pip install pandas scikit-learn nltk symspellpy
   ```
4. **Download NLTK data**

   ```bash
   python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

   Or run the main module:

   ```bash
   python main.py
   ```

## Usage

### Quick Start

```python
from config import get_merged_dataframe
from cleaning import clean_tweets
from vectorizer import count_vectorize, tfidf_vectorize
from train import train_model, evaluate_model

# Load and merge datasets
df = get_merged_dataframe(
    './data/processedNegative.csv',
    './data/processedPositive.csv',
    './data/processedNeutral.csv'
)

# Vectorize tweets
vectorizer, bow_matrix = count_vectorize(df, 'tweet')

# Train a classifier
model = train_model(X_train, y_train)

# Evaluate the model
accuracy = evaluate_model(model, X_test, y_test)
```

### Using Different Vectorizers

```python
from vectorizer import count_vectorize, binary_vectorize, tfidf_vectorize

# Count Vectorizer (word frequencies)
vectorizer, matrix = count_vectorize(df, 'tweet')

# Binary Vectorizer (word presence/absence)
vectorizer, matrix = binary_vectorize(df, 'tweet')

# TF-IDF Vectorizer (term frequency-inverse document frequency)
vectorizer, matrix = tfidf_vectorize(df, 'tweet')
```

### Using Different Tokenization Methods

```python
from tokenizer import stem_tokens, lemmatize_tokens, misspell_and_lemmatize_tokens
from vectorizer import count_vectorize

# With stemming
vectorizer, matrix = count_vectorize(df, 'tweet', tokenizer=stem_tokens)

# With lemmatization
vectorizer, matrix = count_vectorize(df, 'tweet', tokenizer=lemmatize_tokens)

# With spell correction + lemmatization
vectorizer, matrix = count_vectorize(df, 'tweet', tokenizer=misspell_and_lemmatize_tokens)
```

## Contributing

Contributions are welcome! To contribute:


1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## Acknowledgements

This project uses the following open-source libraries:

* [NLTK](https://www.nltk.org/) - Natural Language Toolkit
* [scikit-learn](https://scikit-learn.org/) - Machine Learning in Python
* [SymSpell](https://github.com/mammothb/symspellpy) - Spelling correction
* [pandas](https://pandas.pydata.org/) - Data manipulation and analysis


