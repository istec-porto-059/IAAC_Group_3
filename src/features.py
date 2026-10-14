# src/features.py
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Remove outliers com base no método da Amplitude Interquartil (IQR).
    Requisito do Módulo 4 para limpeza de valores extremos[cite: 1].
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)  # 1º Quartil (25%)
            Q3 = df_clean[col].quantile(0.75)  # 3º Quartil (75%)
            IQR = Q3 - Q1                     # Amplitude Interquartil

            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR

            # Mantém apenas as linhas dentro dos limites aceitáveis
            df_clean = df_clean[(df_clean[col] >= limite_inferior) & (
                df_clean[col] <= limite_superior)]
    return df_clean


def balance_data_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Aplica o algoritmo SMOTE (Synthetic Minority Over-sampling Technique)
    para corrigir o desequilíbrio de classes (Imbalanced Data)[cite: 1].
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(
        f"Distribuição original das classes: {pd.Series(y_train).value_counts().to_dict()}")
    print(
        f"Distribuição após SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")

    return X_resampled, y_resampled
