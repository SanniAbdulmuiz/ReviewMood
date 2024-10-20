from predict import predict_review

if __name__ == "__main__":
    while True:
        review = input("Введите отзыв о фильме (или 'exit' для выхода): ")
        if review.lower() == 'exit':
            break
        prediction = predict_review(review)
        print(f"Тональность отзыва: {prediction}")
