from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(X_train, X_test, max_features=5000):
    """Transforma os textos em matrizes numéricas TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer