# Key Findings — Disneyland Sentiment Analysis

Author: Vishvi
Date: 2026-06-10

---

## What the Data Showed

42,000+ TripAdvisor reviews across three Disneyland parks:
California, Paris, and Hong Kong.

---

## Sentiment Distribution

| Sentiment | Percentage | Insight |
|---|---|---|
| Positive | 70.2% | Majority of visitors love Disneyland |
| Neutral | 14.1% | Satisfied but not thrilled |
| Negative | 15.7% | Pain points exist and are loud |

---

## Park by Park Breakdown

| Park | Positive % | Negative % | Top Complaint |
|---|---|---|---|
| California | 73.4% | 12.1% | Crowds and pricing |
| Paris | 68.9% | 17.8% | Weather and staff |
| Hong Kong | 65.1% | 21.3% | Queue times and food |

Hong Kong had the most negative reviews of all three parks.

---

## Top Positive Triggers

Words most associated with positive reviews:
- magical, amazing, wonderful
- staff friendly, helpful staff
- great ride, fast pass
- beautiful, perfect, loved

---

## Top Negative Triggers

Words most associated with negative reviews:
- long queue, wait time, long line
- expensive food, overpriced
- disappointing, terrible, awful
- crowded, packed

---

## What Bigrams Added

Using ngram_range=(1,2) captured important two-word phrases:
- not good vs good alone
- very happy vs happy alone
- long wait vs wait alone

This improved accuracy by 3 percent over unigrams.

---

## Business Recommendations

| Problem | Recommendation |
|---|---|
| Long queues | FastPass system + virtual queuing |
| Food pricing | Budget meal bundles |
| Hong Kong negativity | Targeted guest experience audit |
| Staff praise | Reward high-rated staff interactions |

---

Built by Vishvi - github.com/vishvi31
