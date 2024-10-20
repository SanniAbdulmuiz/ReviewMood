import pickle
from preprocess import preprocess_text


def predict_review(review_text):
    with open('movie_review_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    cleaned_text = preprocess_text(review_text)
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    return 'Positive' if prediction == 1 else 'Negative'
