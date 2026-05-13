<div align="center">

# 🏰 Disneyland Reviews — Sentiment Analysis using NLP

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/NLP-TF--IDF-FF6B6B?style=for-the-badge"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Complete-00C851?style=for-the-badge"/>

**An end-to-end NLP project performing sentiment analysis on 42,000+ TripAdvisor reviews of Disneyland parks — using TF-IDF vectorisation and Logistic Regression, deployed with joblib.**

*Author: Vishvi · Data Science & AI Practitioner · IBM Professional Certificate (Coursera)*

</div>

---

## 📌 Project Overview

Disneyland parks receive thousands of visitor reviews every day. This project applies Natural Language Processing to automatically classify those reviews as **Positive**, **Neutral**, or **Negative** — giving park management actionable insight into guest sentiment at scale.

> **Goal:** Build a robust text classification pipeline that accurately predicts sentiment from raw review text.

---

## 📊 Dataset

| Attribute | Details |
|---|---|
| **Source** | [Kaggle — Disneyland Reviews (arushchillar)](https://www.kaggle.com/datasets/arushchillar/disneyland-reviews) |
| **Reviews** | 42,000+ TripAdvisor reviews |
| **Parks Covered** | Disneyland California, Paris, Hong Kong |
| **Target** | Sentiment: Positive / Neutral / Negative |
| **Text Column** | `Review_Text` |

---

## 🔬 Project Pipeline

```
Raw Text → EDA → Cleaning → NLP Preprocessing → TF-IDF → Logistic Regression → Deployment
```

### Level 1 — 🔍 Exploratory Data Analysis
- Review length distributions across parks
- Rating breakdowns and sentiment class distribution
- Most frequent words per sentiment class
- Park-wise sentiment comparison

### Level 2 — 🧹 Data Cleaning
- Removed nulls, duplicates, and malformed entries
- Stripped HTML tags, special characters, URLs
- Lowercased all text

### Level 3 — ⚙️ NLP Preprocessing
- Tokenisation
- Stopword removal (NLTK)
- Lemmatisation (WordNetLemmatizer)
- Custom Disneyland-specific stopword list

### Level 4 — 🔢 Feature Engineering
- TF-IDF Vectorisation (`max_features=10,000`, `ngram_range=(1,2)`)
- Bigrams captured contextual phrases like *"not good"*, *"very happy"*

### Level 5 — 🤖 Modelling

| Model | Accuracy | F1-Score (Weighted) |
|---|---|---|
| Logistic Regression (Baseline) | 0.81 | 0.80 |
| **Logistic Regression (Tuned) ✅** | **0.85** | **0.84** |

### Level 6 — 💾 Deployment
- Model and vectoriser serialised using `joblib`
- Reusable inference function for new review text

---

## 📈 Key Results

<div align="center">

| Metric | Score |
|---|---|
| **Accuracy** | **0.85** |
| **F1-Score (Weighted)** | **0.84** |
| **Precision** | 0.85 |
| **Recall** | 0.85 |

</div>

---

## 🔑 Key Findings

1. **Positive reviews dominate** — ~70% of reviews are positive across all parks
2. **Hong Kong Disneyland** has the highest proportion of negative reviews
3. **Staff friendliness** and **ride quality** are the top positive sentiment drivers
4. **Queue times** and **food pricing** are the top negative sentiment triggers
5. **Bigrams improved accuracy** by 3% over unigrams alone — context matters in sentiment

---

## 💡 Business Recommendations

| Finding | Recommendation |
|---|---|
| Long queues drive negative sentiment | Invest in queue management & FastPass systems |
| Food pricing complaints are frequent | Introduce budget-friendly meal bundles |
| Staff praise boosts positive reviews | Recognise and reward high-rated staff interactions |
| Hong Kong park has most negative reviews | Conduct targeted guest experience audit |

---

## 🗂️ Repository Structure

```
Disneyland-Review-Sentiment-Analysis-using-NLP/
│
├── 📓 disneyland_sentiment.ipynb   # Full 6-level pipeline notebook
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📁 data/
│   └── DisneylandReviews.csv       # Raw dataset (from Kaggle)
└── 📁 models/
    ├── sentiment_model.pkl         # Trained Logistic Regression model
    └── tfidf_vectoriser.pkl        # Fitted TF-IDF vectoriser
```

---

## 🚀 Quickstart

```bash
# Clone the repo
git clone https://github.com/vishvi31/Disneyland-Review-Sentiment-Analysis-using-NLP.git
cd Disneyland-Review-Sentiment-Analysis-using-NLP

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook disneyland_sentiment.ipynb
```

---

## 🔮 Predict on New Reviews

```python
import joblib

model = joblib.load("models/sentiment_model.pkl")
vectoriser = joblib.load("models/tfidf_vectoriser.pkl")

def predict_sentiment(review_text):
    features = vectoriser.transform([review_text])
    prediction = model.predict(features)[0]
    return prediction

# Example
print(predict_sentiment("The rides were absolutely magical and the staff were so friendly!"))
# Output: Positive

print(predict_sentiment("Waited 3 hours in the queue. Food was overpriced and cold."))
# Output: Negative
```

---

## 🧰 Tech Stack

- **Python 3.8+**
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualisation
- **nltk** — tokenisation, stopwords, lemmatisation
- **scikit-learn** — TF-IDF, Logistic Regression, evaluation
- **joblib** — model serialisation
- **Jupyter Notebook** — interactive development

---

## 👩‍💻 About the Author

**Vishvi** — Data Science & AI Practitioner, transitioning from an English Literature background (BA Hons, University of Delhi). Currently completing the **IBM Data Science Professional Certificate** on Coursera.

> *"My background in literature gives me a natural edge in NLP — understanding language isn't just technical, it's contextual."*

[![GitHub](https://img.shields.io/badge/GitHub-vishvi31-181717?style=flat&logo=github)](https://github.com/vishvi31)

---

<div align="center">
<sub>Built with 🧠 and 📊 · Part of Vishvi's Data Science Portfolio</sub>
</div>
