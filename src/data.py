# src/data.py
import os
import pandas as pd


def load_raw_data(data_path: str = "data/") -> pd.DataFrame:
    """
    Procura e carrega automaticamente o ficheiro CSV localizado na pasta 'data/'.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"A pasta '{data_path}' não foi encontrada.")

    # Procura por ficheiros .csv dentro da pasta data/
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError(
            f"Nenhum ficheiro .csv encontrado na pasta '{data_path}'. "
            "Certifica-te de que o teu dataset (.csv) está guardado dentro da pasta data/."
        )

    # Carrega o primeiro ficheiro CSV encontrado na pasta
    file_path = os.path.join(data_path, csv_files[0])
    print(f"Dataset encontrado: {file_path}")

    df = pd.read_csv(file_path)
    return df
