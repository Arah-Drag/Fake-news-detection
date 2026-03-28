# 🔍 Fake News Detection System using NLP

A complete end-to-end Machine Learning project that classifies news articles as **REAL** or **FAKE** using Natural Language Processing techniques.

---

## 📁 Project Structure

```
fake_news_project/
│
├── data/
│   ├── Fake.csv          ← Download from Kaggle
│   └── True.csv          ← Download from Kaggle
│
├── src/
│   ├── preprocessing.py  ← Data loading + text cleaning
│   ├── model.py          ← TF-IDF + model training + save/load
│   └── evaluation.py     ← Metrics, confusion matrix, comparison
│
├── models/               ← Auto-created after running main.py
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression.pkl
│   └── naive_bayes.pkl
│
├── plots/                ← Auto-created — all EDA and result charts
│
├── app.py                ← Streamlit web app (Bonus)
├── main.py               ← Full end-to-end pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone / download the project

```bash
cd fake_news_project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Go to → https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Download and place **Fake.csv** and **True.csv** inside the `data/` folder.

---

## 🚀 Run the Pipeline

```bash
python main.py
```

What it does:
1. Loads and labels Fake + Real news CSVs
2. Cleans and preprocesses all text (lowercase, remove stopwords, lemmatize)
3. Generates EDA plots + word clouds (saved in `plots/`)
4. Vectorises text with TF-IDF (50 000 features, unigrams + bigrams)
5. Trains **Logistic Regression** and **Naïve Bayes**
6. Evaluates both models (accuracy, precision, recall, F1, confusion matrix)
7. Compares models side-by-side
8. Saves artefacts to `models/`

---

## 🌐 Launch the Web App (Bonus)

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.  
Paste any news article → click **Analyse Article** → get **FAKE / REAL** prediction with confidence score.

---

## 📊 Expected Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | ~98–99 % | ~98–99 % | ~98–99 % | ~98–99 % |
| Naïve Bayes | ~94–96 % | ~94–96 % | ~94–96 % | ~94–96 % |

---

## 🧪 NLP Pipeline

```
Raw text
  ↓  lowercase
  ↓  remove URLs
  ↓  remove punctuation & digits
  ↓  tokenise
  ↓  remove stopwords (NLTK English list)
  ↓  lemmatize (WordNet)
  ↓  TF-IDF vectorisation (unigrams + bigrams)
  ↓  ML model → FAKE / REAL
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| pandas, numpy | Data handling |
| scikit-learn | TF-IDF, models, metrics |
| nltk | Stopwords, lemmatization |
| matplotlib, seaborn | Plots |
| wordcloud | Word cloud images |
| joblib | Model serialisation |
| streamlit | Web app |

---

## 👤 Author

Built as part of an NLP / ML coursework project.  
Dataset credit: [Clément Bisaillon on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
