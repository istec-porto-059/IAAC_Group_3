import re
import pandas as pd

def clean_text(text: str) -> str:
    """Limpa o texto do e-mail removendo pontuações, URLs e caracteres especiais."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'http\S+|www\S+|<.*?>', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_and_preprocess_data(file_path: str, n_normal_samples: int = 10000) -> pd.DataFrame:
    """Carrega o dataset, realiza amostragem balanceada e aplica a limpeza no texto."""
    df = pd.read_csv(file_path)
    
    df_fraud = df[df['Label'] == 1]
    df_normal = df[df['Label'] == 0].sample(n=n_normal_samples, random_state=42)
    
    df_subset = pd.concat([df_fraud, df_normal]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_subset['clean_body'] = df_subset['Body'].apply(clean_text)
    
    return df_subset