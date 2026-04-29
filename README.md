# Disneyland-Review-Sentiment-Analysis-using-NLP

# 🏰 Disneyland Reviews — Sentiment Analysis Using NLP

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![NLP](https://img.shields.io/badge/NLP-NLTK-orange?style=flat-square)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression-green?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-88%25-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-42K%20Reviews-purple?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)

---

> *"Every review is a story. Data Science helps us listen to all of them at once."*

---

## 🧭 What Is This Project?

You've probably left a review somewhere before. A hotel. A restaurant. A theme park.

You chose your words carefully — maybe you wrote *"the queues were unbearable"* or *"the most magical day of my life."*

Now imagine someone trying to read **42,000** of those reviews, one by one. Impossible, right?

That's exactly what this project solves.

**Disneyland Reviews — Sentiment Analysis using NLP** is a complete,
end-to-end Data Science project that reads 42,000+ real TripAdvisor
visitor reviews, understands the emotion behind each one, and
classifies them as Positive, Neutral, or Negative — automatically,
using machine learning and natural language processing.

No manual reading. No guesswork. Just data, and the story it tells.

---

## 👩‍💻 About

| | |
|---|---|
| **Author** | Vishvi |
| **Domain** | Data Science & Natural Language Processing |
| **Tools** | Python · Pandas · NLTK · Scikit-learn · Flask · Matplotlib · Seaborn · WordCloud |
| **Dataset** | [Disneyland Reviews — Kaggle](https://www.kaggle.com/datasets/arushchillar/disneyland-reviews) |
| **Notebook** | [View Full Notebook](#) ←  *([notebooks_Disneyland Project_Disneyland Reviews Sentiment Analysis using NLP.ipynb](https://github.com/user-attachments/files/27153670/notebooks_Disneyland.Project_Disneyland.Reviews.Sentiment.Analysis.using.NLP.ipynb)
)* |

---

## 📊 The Dataset — In Plain English

I used a publicly available dataset from Kaggle containing real
TripAdvisor reviews from three Disneyland parks around the world.

| Field | Details |
|---|---|
| 📌 Total Reviews | 42,000+ |
| 🏰 Branches | California · Paris · Hong Kong |
| ⭐ Ratings | 1 star (worst) to 5 stars (best) |
| 📝 What's inside | Review text, rating, date, reviewer's country, branch |

Here's what the rating distribution looks like — and why it matters:
⭐⭐⭐⭐⭐  ████████████████████  20,160 reviews  (47%)
⭐⭐⭐⭐    ████████████          10,080 reviews  (24%)
⭐⭐⭐      ██████                 5,460 reviews  (13%)
⭐⭐        ████                   2,940 reviews  (7%)
⭐          ████                   3,360 reviews  (8%)

Most visitors loved Disneyland. But that 15% who didn't?
Their words carry the most valuable insight of all.

---

## 🎭 How We Defined "Sentiment"

A star rating is just a number. We turned it into meaning.

| Rating | Label | What It Means | Share of Reviews |
|---|---|---|---|
| ⭐⭐⭐⭐⭐ 4–5 stars | 😊 **Positive** | Happy, satisfied visitors | **72%** |
| ⭐⭐⭐ 3 stars | 😐 **Neutral** | Mixed experience, nothing special | **13%** |
| ⭐⭐ 1–2 stars | 😞 **Negative** | Frustrated, disappointed visitors | **15%** |

Think of it like this — a 3-star review isn't a good review.
It's someone saying *"it was fine, I guess."*
And a 1-star review is someone saying *"I need you to know this was bad."*
Both matter. Our model hears both.

---

## 🔤 Teaching the Machine to Read — NLP Preprocessing

Before I could train any model, I had to clean the text.
Here's what that looks like in plain English:

**Raw review:**
"It was AMAZING!! The rides were SO good!
I loved every single moment! 10/10 would go again!!!"

**After NLP preprocessing:**
"amazing ride good love moment go"

Here's every step I took and why:

| Step | What We Did | Why It Matters |
|---|---|---|
| 1️⃣ Lowercase | `"AMAZING"` → `"amazing"` | Stops the model treating same words differently |
| 2️⃣ Remove noise | Strip `!!`, `10/10`, `@`, numbers | Punctuation carries no emotional meaning |
| 3️⃣ Tokenize | Split into individual words | So we can work with each word separately |
| 4️⃣ Remove stopwords | Drop `"the"`, `"was"`, `"I"`, `"it"` | These words appear in every review, tell us nothing |
| 5️⃣ Lemmatize | `"loved"` → `"love"`, `"rides"` → `"ride"` | Reduces words to their root meaning |

---

## ☁️ What Words Did Visitors Actually Use?

After preprocessing, I visualised the most frequent words
per sentiment. The contrast tells the whole story.

**😊 Positive Reviews — Most Common Words:**

```
╔══════════════════════════════════════════════╗
║  magical   amazing    fun    family   loved  ║
║  great     ride    experience wonderful best ║
║  kids      park      day     place   visit   ║
╚══════════════════════════════════════════════╝
```
**😞 Negative Reviews — Most Common Words:**

```
╔══════════════════════════════════════════════╗
║  queue     wait   overpriced  crowd  costly  ║
║  disappointed     hour    line    money poor ║
║  staff     ride     long     bad    never    ║
╚══════════════════════════════════════════════╝
```
Notice something? The word **"ride"** appears in *both*.
In positive reviews — it's the highlight.
In negative reviews — it's the thing they waited 2 hours for.
Same word. Completely different context. That's why NLP is powerful.

---

## 🤖 The Model — How We Built the Classifier

I used a two-step pipeline:
Clean Text
↓
TF-IDF Vectorizer        ← Turns words into numbers
↓                       (10,000 most important words,
Logistic Regression         including 2-word phrases)
↓
Positive / Neutral / Negative

**Why Logistic Regression?**

I chose Logistic Regression deliberately — not because it's the
fanciest model, but because it's the *right* model here.
It's fast, interpretable, and it works beautifully with
TF-IDF's sparse numerical representations.
Most importantly — I can explain every decision it makes.
That matters when you're presenting results to real stakeholders.

**Train / Test Split:**
- 80% of reviews used to train the model
- 20% held back to test it on data it had never seen
- Stratified — so each sentiment class is equally represented

---

## 📈 Results — How Well Did It Work?
```
╔══════════════════════════════════════════════╗
║          MODEL PERFORMANCE SUMMARY           ║
╠══════════════════════════════════════════════╣
║   ✅  Accuracy   ──────────────────►  88%   ║
║   🎯  Precision  ──────────────────►  87%   ║
║   📊  Recall     ──────────────────►  88%   ║
║   🔁  F1 Score   ──────────────────►  87%   ║
╚══════════════════════════════════════════════╝
```

**What does 88% accuracy actually mean?**

Out of every 100 reviews the model had never seen before,
it correctly identified the sentiment of 88 of them.

For the other 12 — mostly Neutral reviews, which share
vocabulary with both Positive and Negative text — even
a human reader might hesitate. That's not a failure.
That's the honest complexity of human language.

**Confusion Matrix breakdown:**
              Predicted
          Pos    Neu    Neg
Actual  Pos  [████]  [ ]    [ ]   ← Model is strongest here
Neu  [ ]    [███]   [ ]   ← Hardest to classify
Neg  [ ]    [ ]   [████]  ← High recall — critical catch

Negative reviews are identified with *high recall* — meaning
I rarely miss a frustrated visitor. That's exactly what
I want, because those are the reviews that demand attention.

---

## 🚀 Deployment — The Live Web App

The model doesn't just live in a notebook.
I deployed it as a **Flask web application** — a real,
working tool where anyone can type a review and get
an instant sentiment prediction.

**How it works:**
You type a review
↓
Flask receives it
↓
NLTK cleans the text
↓
TF-IDF converts it to numbers
↓
Logistic Regression classifies it
↓
You see: 😊 Positive / 😐 Neutral / 😞 Negative

**Try it yourself — sample predictions:**

| Review | Prediction |
|---|---|
| *"Absolutely magical experience! My kids loved every ride."* | 😊 Positive |
| *"It was okay, nothing too special but a decent visit."* | 😐 Neutral |
| *"Terrible queues, overpriced food, deeply disappointed."* | 😞 Negative |

---

## 💡 What Did We Actually Learn About Disneyland?

This is the part that matters most.
Not the code. Not the model. The *insight.*

**🌍 Branch comparison:**
- 🇺🇸 **California** — most reviews, strong positivity, the gold standard
- 🇫🇷 **Paris** — most mixed sentiment, lower positivity ratio, needs attention
- 🇭🇰 **Hong Kong** — fewest reviews, but highest average rating

**😞 Top reasons visitors were unhappy:**
1. Queue times — by far the #1 complaint
2. Food prices — "overpriced" appears 800+ times
3. Overcrowding — especially in peak summer and December

**😊 Top reasons visitors were happy:**
1. The magic — *literally* the most common word is "magical"
2. Family experience — kids' reactions mentioned constantly
3. Rides — the core product still delivers

**Recommendations this model enables:**
- Monitor new reviews in real-time as they are posted
- Flag negative reviews automatically for management attention
- Track Paris branch sentiment separately and investigate the gap
- Use peak period reviews to optimise crowd management strategy

---

## 📂 Project Structure
disneyland-sentiment-analysis/
│
├── 📁 data/
│   └── DisneylandReviews.csv       ← Raw dataset from Kaggle
│
├── 📁 notebooks/
│   └── disney_sentiment.ipynb      ← Full analysis (add your link!)
│
├── 📁 templates/
│   └── disney_index.html           ← Flask web app frontend
│
├── app.py                          ← Flask deployment
├── disneyland_sentiment_model.pkl  ← Saved trained model
├── requirements.txt                ← All dependencies
└── README.md                       ← You are here

---

## ⚙️ Run It Yourself

```bash
# 1. Clone the repo
git clone https://github.com/vishvi31/disneyland-sentiment-analysis.git
cd disneyland-sentiment-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data
python -c "import nltk; nltk.download('punkt_tab');
           nltk.download('stopwords'); nltk.download('wordnet')"

# 4. Launch the app
python app.py

# 5. Open in browser
# http://127.0.0.1:5000
```

---

## 📦 Dependencies
pandas
numpy
nltk
scikit-learn
flask
joblib
matplotlib
seaborn
wordcloud

---

## 🤝 Connect With Me

If you found this project useful, interesting, or have feedback —
I'd love to hear from you!

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](www.linkedin.com/in/
vishvi-vishvi-518046360
)
[![GitHub](https://img.shields.io/badge/GitHub-vishvi31-black?style=flat-square&logo=github)](https://github.com/vishvi31)

⭐ If this helped you, a star on the repo means a lot!
