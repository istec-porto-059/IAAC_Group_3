# main.py
from src.data import load_raw_data
from src.features import remove_outliers_iqr, balance_data_smote
from src.training import train_and_tune_all_models
from src.evaluation import evaluate_all_models
from sklearn.model_selection import train_test_split


def main():
    print("1. A carregar os dados...")
    df = load_raw_data()

    print("2. Módulo 4: Limpeza de Outliers...")
    numeric_cols = df.select_dtypes(
        include=['float64', 'int64']).columns.tolist()
    if 'target' in numeric_cols:
        numeric_cols.remove('target')

    df_clean = remove_outliers_iqr(df, columns=numeric_cols)

    X = df_clean.drop(columns=['target'])
    y = df_clean['target']

    print("3. Divisão Treino / Teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("4. Módulo 4: Balanceamento de dados com SMOTE...")
    X_train_balanced, y_train_balanced = balance_data_smote(X_train, y_train)

    print("5. Módulo 6 & 7: Treino e Hyperparameter Tuning dos 7 modelos...")
    best_models = train_and_tune_all_models(X_train_balanced, y_train_balanced)

    print("6. Módulo 7: Avaliação e geração de relatórios...")
    evaluate_all_models(best_models, X_test, y_test)


if __name__ == "__main__":
    main()
