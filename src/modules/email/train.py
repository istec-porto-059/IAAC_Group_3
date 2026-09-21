import os
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score

from src.modules.email.data_prep import load_and_preprocess_data
from src.modules.email.features import build_tfidf_features

# Garante que os caminhos apontam para a raiz do projeto (IAAC_Group_3)
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "data" / "raw" / "enron_data_fraud_labeled.csv"
MODELS_DIR = BASE_DIR / "models"

def run_pipeline():
    print("1. Carregando e pré-processando dados...")
    df = load_and_preprocess_data(str(DATA_PATH), n_normal_samples=10000)
    
    print("2. Divisão treino/teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_body'], df['Label'], test_size=0.20, random_state=42, stratify=df['Label']
    )
    
    print("3. Extraindo features TF-IDF...")
    # Corrigido de create_tfidf_features para build_tfidf_features
    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_features(X_train, X_test)
    
    print("4. Treinando e comparando múltiplos modelos...")
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100, n_jobs=-1)
    }
    
    best_model = None
    best_model_name = ""
    best_recall = 0.0
    
    for name, model in models.items():
        print(f"\n=== Avaliando: {name} ===")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fraude (1)']))
        
        rec = recall_score(y_test, y_pred, pos_label=1)
        if rec > best_recall:
            best_recall = rec
            best_model = model
            best_model_name = name
            
    print(f"\n>> Melhor Modelo para Cibersegurança: {best_model_name} (Recall Fraude: {best_recall:.4f})")
    
    print("5. Salvando artefatos na pasta models/...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")
    print("Pipeline concluído com sucesso!")

if __name__ == "__main__":
    run_pipeline()