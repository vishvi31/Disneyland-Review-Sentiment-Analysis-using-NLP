# NLP Pipeline — Step by Step

Author: Vishvi
Date: 2026-06-10

---

## Overview

This file documents every step of the NLP pipeline
used in the Disneyland Sentiment Analysis project.
Each step is explained in plain English with code.

---

## Step 1 — Load the Data

```python
import pandas as pd

df = pd.read_csv('data/DisneylandReviews.csv', encoding='latin-1')
print(df.shape)
print(df.columns.tolist())
print(df['Rating'].value_counts())
print(df['Review_Text'].head(3))
```

---

## Step 2 — Create Sentiment Labels

```python
def label_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

df['Sentiment'] = df['Rating'].apply(label_sentiment)
print(df['Sentiment'].value_counts())
```

---

## Step 3 — Clean the Text

```python
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Cleaned'] = df['Review_Text'].apply(clean_text)
print(df['Cleaned'].head(3))
```

---

## Step 4 — NLP Preprocessing

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()

custom_stops = {'disneyland', 'disney', 'park', 'ride', 'place'}
stop_words.update(custom_stops)

def preprocess(text):
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df['Processed'] = df['Cleaned'].apply(preprocess)
print(df['Processed'].head(3))
```

---

## Step 5 — TF-IDF Vectorisation

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectoriser = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)
X = vectoriser.fit_transform(df['Processed'])
print('Feature matrix shape:', X.shape)
```

---

## Step 6 — Train and Evaluate

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## Step 7 — Save the Model

```python
import joblib

joblib.dump(model,     'models/sentiment_model.pkl')
joblib.dump(vectoriser,'models/tfidf_vectoriser.pkl')
print('Model and vectoriser saved!')
```

---

Built by Vishvi - github.com/vishvi31
