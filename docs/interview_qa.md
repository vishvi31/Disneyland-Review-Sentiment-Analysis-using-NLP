# Interview Q&A — Disneyland NLP Project

Author: Vishvi
Date: 2026-06-10

Likely questions a recruiter or interviewer might ask
about this project — with clear answers.

---

## Q1: Why did you choose Logistic Regression over other models?

Logistic Regression works extremely well for text classification
because TF-IDF creates high-dimensional sparse features.
LR is fast, interpretable, and performs on par with
more complex models on this type of data.
I also tried Naive Bayes and SVM but LR gave the best F1.

---

## Q2: What is TF-IDF and why did you use it?

TF-IDF stands for Term Frequency - Inverse Document Frequency.
TF = how often a word appears in one review.
IDF = how rare that word is across all reviews.
Words like 'the' appear everywhere so IDF makes their score low.
Words like 'magical' are rare and meaningful so score high.
I used it because it captures importance of words, not just frequency.

---

## Q3: Why did you use bigrams?

Unigrams miss context. 'not good' as two separate words
looks like it contains 'good' which is positive.
Bigrams capture 'not good' as one unit, which is negative.
This improved accuracy by 3 percent in my experiments.

---

## Q4: How did you handle class imbalance?

The dataset had 70 percent positive, 15 percent neutral,
and 15 percent negative reviews.
I used class_weight='balanced' in Logistic Regression
which automatically adjusts weights inversely proportional
to class frequencies. This improved recall for minority classes.

---

## Q5: What is lemmatisation and why use it over stemming?

Stemming chops word endings crudely. 'running' becomes 'runn'.
Lemmatisation maps words to their true root. 'running' becomes 'run'.
I used lemmatisation because it produces real words
which makes the TF-IDF vocabulary cleaner and more meaningful.

---

## Q6: What would you improve if given more time?

1. Try BERT or DistilBERT for better contextual understanding
2. Add aspect-based sentiment (food vs rides vs staff separately)
3. Build a live Flask web app for real-time predictions
4. Collect more neutral class samples to balance the dataset
5. Add confidence scores to predictions

---

## Q7: What was your biggest challenge?

The neutral class was hardest to classify correctly.
Reviews like 'It was okay' or 'Nothing special' sit between
positive and negative and the model often misclassified them.
I addressed this with class_weight='balanced' and stratified splits.

---

Built by Vishvi - github.com/vishvi31
