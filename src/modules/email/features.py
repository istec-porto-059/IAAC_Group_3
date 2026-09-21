import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_features(X_train, X_test, max_features=5000):
    """
    Recebe as séries de texto X_train e X_test e ajusta o vetorizador TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    
    # Treina o vetorizador no treino e transforma ambos os conjuntos
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, vectorizer