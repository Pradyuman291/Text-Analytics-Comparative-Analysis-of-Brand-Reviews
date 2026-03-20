# Text-Analytics-Comparative-Analysis-of-Brand-Reviews
Data Mining — predicting life insurance purchase likelihood across 25,271 customers using Logistic Regression, KNN &amp; Random Forest in Python.
# Text Analytics — Comparative Analysis of Brand Reviews

**Module:** MGT7216 – Data Mining  
**Author:** Pradyuman Kumar 
**Supervisor:** Dr. V. Charles

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Recommendations](#recommendations)
- [How to Run](#how-to-run)
- [References](#references)

---

## Project Overview

This project applies **Natural Language Processing (NLP) and emotion analytics** to compare customer brand reviews for two competing brands — Brand H and Brand Z. Using a combination of supervised and semi-supervised machine learning classifiers, the project aims to:

- Automatically detect the emotion expressed in customer reviews (joy, surprise, anger, disgust, fear, sadness, neutral)
- Perform sentiment and polarity analysis on review text
- Deliver comparative brand insight to inform business strategy

The dataset contained **5,722 records**, of which only **627 had labelled emotions** — making this a challenging real-world NLP problem requiring semi-supervised techniques.

---

## Dataset Description

| Field | Description |
|---|---|
| ID | Unique record identifier |
| brand_name | Brand name (H or Z) |
| country | Country of the reviewer |
| star_rating | Customer star rating (1–5) |
| emotions | Emotion label (627 labelled; 5,095 blank/NaN) |
| text_reviews | Raw customer review text |
| filtered_reviews | Cleaned review text (generated during preprocessing) |

**Dataset statistics:**
- Total records: 5,722
- Labelled emotion records: 627 (10.9%)
- Unlabelled records: 5,095 (89.1%)
- Emotion classes: anger, disgust, fear, joy, neutral, sadness, surprise

*Note: The dataset was provided as part of the MSc programme at Queen's University Belfast and is not included in this repository. The code and analysis are original work.*

---

## Methodology

### Text Pre-Processing Pipeline

All raw review text is passed through the following cleaning steps before being fed into any model:

```
Raw Text
   │
   ├── 1. Remove URLs & HTML tags          (re.sub)
   ├── 2. Lowercase conversion             (str.lower)
   ├── 3. Remove 1–2 letter words          (re.sub r'\b\w{1,2}\b')
   ├── 4. Remove non-letter characters     (re.sub r'[^a-z\s]')
   ├── 5. Remove punctuation               (string.punctuation)
   ├── 6. Tokenization                     (nltk word_tokenize)
   ├── 7. Stop word removal                (nltk stopwords)
   └── 8. Lemmatization                    (WordNetLemmatizer)
```

**Why Lemmatization over Stemming?** Lemmatization preserves linguistic context and returns valid root words (e.g., "running" → "run"), producing higher-quality features for downstream classification.

Text is then vectorised using **TF-IDF** (`TfidfVectorizer`) before being passed to classifiers.

---

### Supervised Learning Models

Seven classifiers trained on 627 labelled samples (80/20 train/test split, `random_state=40425764`):

| # | Classifier | Accuracy |
|---|---|---|
| 1 | Logistic Regression | 54.76% |
| 2 | Decision Tree | 54.76% |
| 3 | Naive Bayes | 35.71% |
| 4 | ✅ Support Vector Classifier (SVC) | **57.93%** |
| 5 | Random Forest | 54.76% |
| 6 | ANN (Neural Network) | 9.52% |
| 7 | KNN (k=5) | 40.47% |

**SVC with a linear kernel achieved the best overall accuracy.**

---

### Semi-Supervised Learning Models

To make use of the 5,095 unlabelled records, two semi-supervised approaches were evaluated:

| Classifier | Micro-Avg F1 Score |
|---|---|
| Supervised SGD Classifier (baseline) | 0.500 |
| Self-Training Classifier (10 iterations) | 0.500 |

The Self-Training Classifier iteratively added new labels each iteration, progressively expanding the training set from labelled data outward.

```python
st_pipeline = Pipeline([
    ("vect", CountVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.8)),
    ("tfidf", TfidfTransformer()),
    ("clf", SelfTrainingClassifier(SVC(kernel='linear', probability=True), verbose=True))
])
```

---

### Sentiment & Polarity Analysis

Sentiment and polarity scores computed using **TextBlob**:

```python
blob = TextBlob(text)
polarity = blob.sentiment.polarity
# polarity > 0 → Positive | polarity < 0 → Negative | polarity == 0 → Neutral
```

---

## Results

### Best Supervised Model: Support Vector Classifier (SVC) — 57.93%

| Emotion | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| anger | 0.33 | 0.08 | 0.12 | 13 |
| disgust | 0.50 | 0.58 | 0.54 | 12 |
| fear | 0.50 | 0.43 | 0.46 | 14 |
| joy | 0.79 | 0.79 | **0.79** | 24 |
| neutral | 0.65 | 0.65 | 0.65 | 23 |
| sadness | 0.50 | 0.57 | 0.55 | 21 |
| surprise | 0.50 | 0.68 | 0.58 | 19 |
| **Overall** | **0.57** | **0.58** | **0.56** | **126** |

The model performs strongest on **"joy"** (F1: 0.79) and weakest on **"anger"** (F1: 0.12).

### Key Visualisations

| Plot | Type | Key Insight |
|---|---|---|
| Emotion vs Star Rating | Violin plot | Joy → 5 stars; Anger/Sadness → ~2 stars |
| Emotion Distribution | Bar chart | Surprise most common; Anger least common |
| Brand Name & Emotions | Grouped bar | Brand H has more joy & surprise than Z |
| Brand Heatmap | Heatmap | H outperforms Z on positive emotions overall |
| Sentiment vs Star Rating | Violin plot | Positive sentiment strongly correlates with 4–5 stars |

---

## Key Findings

- **SVC is the best emotion classifier at 57.93%** — the 627-record labelled dataset is the primary limiting factor
- **Joy and Surprise** are most associated with 5-star reviews — brands should consistently replicate whatever drives these
- **Anger** was the hardest emotion to predict across all models — a known challenge in short-text NLP
- **Brand H outperforms Brand Z** in overall positive sentiment, joy and surprise; Z has a higher proportion of neutral and negative reviews
- Semi-supervised approaches plateau at 50% — transformer-based models (BERT, RoBERTa) could unlock significantly higher performance

---

## Recommendations

- **Standardise what works** — both brands should identify the specific practices that generate joy and surprise reviews and make these consistent across all touchpoints
- **Focus Brand Z on improvement** — Z's reviews cluster around neutral; small improvements in customer experience could shift sentiment measurably positive
- **Invest in more emotion labelling** — labelling even 500–1,000 additional records would significantly improve all model accuracies
- **Explore transformer-based models** — replacing TF-IDF + SVC with fine-tuned BERT or RoBERTa could push accuracy well beyond the 60% ceiling
- **Integrate into CRM pipelines** — real-time emotion tagging of incoming reviews could trigger automatic customer service escalation for negative emotions

---

## How to Run

### Prerequisites
Python 3.8+ required. Developed in Google Colab but can be run locally.

### Installation

```bash
git clone https://github.com/Pradyuman291/brand-review-text-analytics.git
cd brand-review-text-analytics
pip install pandas numpy matplotlib seaborn scikit-learn nltk textblob openpyxl
python -m nltk.downloader punkt stopwords wordnet vader_lexicon
```

### Running the Code

**Option A — Google Colab (Recommended)**
1. Open Google Colab
2. Upload `brand_review_analytics.ipynb`
3. Mount Google Drive and update the dataset path
4. Run all cells (Runtime > Run all)

**Option B — Local Jupyter Notebook**
```bash
pip install jupyter
jupyter notebook notebooks/brand_review_analytics.ipynb
```

**Option C — Python Script**
```bash
python src/main.py
```

---

## Technologies Used

- **Language:** Python
- **Libraries:** `scikit-learn`, `nltk`, `TextBlob`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `openpyxl`
- **Environment:** Google Colab / Jupyter Notebook

---

## References

- Nasır, S. (2017). Customer retention strategies and customer loyalty. IGI Global eBooks, pp. 1178–1201.
- Xiang, Z. et al. (2017). A comparative analysis of major online review platforms. *Tourism Management*, 58, pp. 51–65.
- Guo, Y. et al. (2022). Mining multi-brand characteristics from online reviews for competitive analysis. *Electronic Commerce Research and Applications*, 53, p. 101141.
- He, Y. and Zhou, D. (2011). Self-training from labeled features for sentiment analysis. *Information Processing & Management*, 47(4), pp. 606–616.
- Hussain, A. and Wang, Z. (2018). Semi-supervised learning for big social data analysis. *Neurocomputing*, 275, pp. 1662–1673.

---

*This project was submitted as academic coursework at Queen's University Belfast (MGT7216 Data Mining). Code is shared for educational reference only.*

Made with 🐍 Python & 🤖 scikit-learn / NLTK / TextBlob | Queen's University Belfast
