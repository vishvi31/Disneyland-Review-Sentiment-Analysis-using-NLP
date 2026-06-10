# Model Card — Disneyland Sentiment Classifier

Author: Vishvi
Date: 2026-06-10

---

## Model Details

| Field | Details |
|---|---|
| Model type | Logistic Regression |
| Task | Multi-class text classification |
| Classes | Positive, Neutral, Negative |
| Vectoriser | TF-IDF (max 10,000 features, bigrams) |
| Library | scikit-learn |
| Serialised with | joblib |

---

## Performance

| Metric | Score |
|---|---|
| Accuracy | 0.85 |
| F1 Weighted | 0.84 |
| Precision | 0.85 |
| Recall | 0.85 |

---

## Training Data

- 42,000+ TripAdvisor Disneyland reviews
- Source: Kaggle (arushchillar)
- Parks: California, Paris, Hong Kong
- 80 percent train / 20 percent test split
- Stratified split to preserve class balance

---

## How to Use

```python
import joblib

model      = joblib.load('models/sentiment_model.pkl')
vectoriser = joblib.load('models/tfidf_vectoriser.pkl')

def predict(text):
    features = vectoriser.transform([text])
    return model.predict(features)[0]

print(predict('Amazing experience! Staff were so friendly!'))
print(predict('Waited 3 hours. Food was cold and overpriced.'))
```

---

## Limitations

- Trained only on Disneyland reviews
- May not generalise to other domains
- Neutral class is hardest to predict correctly
- Does not handle sarcasm well

---

## Ethical Considerations

This model is for educational and portfolio purposes only.
It should not be used to make business decisions without
further validation on a larger, more diverse dataset.

---

Built by Vishvi - github.com/vishvi31
