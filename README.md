# 🎬 Movie Review Sentiment Analysis

A machine learning project that classifies IMDb movie reviews as **positive** or **negative**. It uses TF-IDF for feature extraction and lets you choose between four models in a simple Streamlit web app.

## Features
- Choose between 4 models: Logistic Regression, Naive Bayes, Linear SVM and Decision Tree
- Enter any movie review and get the predicted sentiment (green for positive, red for negative)
- Confidence score for the prediction (where the model supports it)
- Compare the predictions of all models on the same review
- Colab notebook with EDA, model comparison and evaluation

## Dataset
[IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (`IMDB Dataset.csv`), with 25,000 positive and 25,000 negative reviews (a few duplicates are removed during preprocessing).

## Project workflow
1. Load the dataset and remove duplicates
2. Exploratory data analysis (class balance, review length, word clouds)
3. Clean the text (remove HTML tags, convert to lowercase)
4. Convert text to numbers using TF-IDF (unigrams + bigrams, stopwords removed)
5. Train and compare 4 models
6. Evaluate using accuracy, precision, recall, F1-score and confusion matrices
7. Save the models and use them in the Streamlit app

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 89.64% |
| Naive Bayes | 86.93% |
| Linear SVM | 89.00% |
| Decision Tree | 73.87% |


## How to run

1. Clone the repository
   ```
   git clone https://github.com/<your-username>/Text_Classification.git
   cd Text_Classification
   ```
2. Install the requirements
   ```
   pip install -r requirements.txt
   ```
3. Start the app
   ```
   streamlit run app.py
   ```

The trained models are already included, so you don't need to train anything.

## Retraining the models
Open `Sentiment_Analysis_IMDb.ipynb` in [Google Colab](https://colab.research.google.com), run all cells and upload `IMDB Dataset.csv` when asked. The notebook downloads three files (`tfidf_vectorizer.joblib`, `all_models.joblib`, `accuracies.joblib`). Put them in the project folder, replacing the old ones.

## Project structure
```
├── app.py                          # Streamlit app
├── Sentiment_Analysis_IMDb.ipynb   # Colab notebook (EDA + training)
├── IMDB Dataset.csv                # Dataset
├── tfidf_vectorizer.joblib         # Saved TF-IDF vectorizer
├── all_models.joblib               # Saved models
├── accuracies.joblib               # Saved test accuracies
├── requirements.txt
└── README.md
```

## Tech stack
Python, pandas, scikit-learn, matplotlib, seaborn, Streamlit
