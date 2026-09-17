import joblib
from src.data_prep import clean_text

def predict_email(text: str):
    """Carrega o modelo salvo e faz a predição para um novo e-mail."""
    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    
    clean = clean_text(text)
    features = vectorizer.transform([clean])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4)
    }

if __name__ == "__main__":
    sample_email = "URGENT: Confidential wire transfer required immediately to proceed with transaction."
    result = predict_email(sample_email)
    print("\n--- Teste de Inferência ---")
    print(f"Texto: {sample_email}")
    print(f"Resultado: {result}")