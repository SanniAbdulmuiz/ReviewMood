import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


# Загрузка модели и векторизатора
with open('model/movie_review_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('model/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Загружаем тестовые данные
df = pd.read_csv('data/data_comment.csv')

# Обработка и векторизация текста
df['review'] = df['review'].apply(preprocess_text)
X_test_tfidf = vectorizer.transform(df['review'])

# Прогнозирование
y_test = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
y_pred = model.predict(X_test_tfidf)

# Оценка результатов
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
