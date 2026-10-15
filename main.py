from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import sys
import glob
import pandas as pd
import numpy as np

# Adicionar a pasta 'src' ao path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'src')))


# Importação dos modelos para o Módulo 6 (Supervised Learning)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def identificar_coluna_target(df):
    """Deteta automaticamente a coluna correspondente à classe alvo no DataFrame."""
    colunas = df.columns.tolist()
    candidatos = ['label', 'spam', 'target', 'prediction',
                  'category', 'class', 'status', 'result', 'type']

    for cand in candidatos:
        for col in colunas:
            if str(col).strip().lower() == cand:
                return col

    # Se não encontrar por nome, usa a última coluna
    return colunas[-1]


def processar_e_treinar_dataset(caminho_ficheiro):
    nome_ficheiro = os.path.basename(caminho_ficheiro)
    print(f"\n==========================================================")
    print(f" A PROCESSAR DATASET: {nome_ficheiro}")
    print(f"==========================================================")

    try:
        df = pd.read_csv(caminho_ficheiro, low_memory=False)
    except Exception as e:
        print(f"[Erro] Não foi possível ler o ficheiro {nome_ficheiro}: {e}")
        return

    print(f"Dimensões originais: {df.shape}")

    # 1. Amostragem para manter a execução rápida e poupar memória RAM
    if len(df) > 30000:
        print("-> A aplicar amostragem de 30.000 linhas para treino otimizado...")
        df = df.sample(n=30000, random_state=42).reset_index(drop=True)

    # 2. Identificar Target
    target_col = identificar_coluna_target(df)
    print(f"-> Coluna alvo (target) detetada: '{target_col}'")

    y = df[target_col]

    # Padronizar target para 0 e 1
    if y.dtype == 'object' or len(np.unique(y)) > 2:
        # Converter para numérico se necessário ou binarizar
        y = pd.factorize(y)[0]
    else:
        y = y.astype(int)

    # 3. Módulo 4: Feature Engineering e Seleção Numérica
    X_num = pd.DataFrame()

    # Se existirem colunas de texto (ex: Subject, Body em emails)
    if 'Subject' in df.columns:
        X_num['subj_len'] = df['Subject'].astype(str).str.len()
    if 'Body' in df.columns:
        X_num['body_len'] = df['Body'].astype(str).str.len()

    colunas_num_existentes = df.drop(
        columns=[target_col], errors='ignore').select_dtypes(include=[np.number])
    X = pd.concat([X_num, colunas_num_existentes], axis=1).fillna(0)

    if X.empty or X.shape[1] == 0:
        X = df.drop(columns=[target_col], errors='ignore').apply(
            pd.to_numeric, errors='coerce').fillna(0)

    print(f"Features selecionadas: {X.shape[1]} colunas")

    # 4. Splitting & Scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Módulo 6: Supervised Learning (Modelos Mínimos M)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "k-NN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Naïve Bayes": GaussianNB(),
        "Linear SVM": LinearSVC(dual=False, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            eval_metric='logloss', random_state=42)

    # 6. Módulo 7: Model Evaluation (Métricas M)
    resultados = []

    for nome, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            auc = np.nan
            if hasattr(model, "predict_proba"):
                try:
                    auc = roc_auc_score(
                        y_test, model.predict_proba(X_test_scaled)[:, 1])
                except:
                    pass
            elif hasattr(model, "decision_function"):
                try:
                    auc = roc_auc_score(
                        y_test, model.decision_function(X_test_scaled))
                except:
                    pass

            resultados.append({
                "Modelo": nome,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                "AUC-ROC": auc
            })
        except Exception as err:
            print(f"   [Aviso] Falha ao treinar {nome}: {err}")

    # Apresentar tabela comparativa para este dataset especificamente
    df_res = pd.DataFrame(resultados).sort_values(
        by="F1-Score", ascending=False).reset_index(drop=True)
    print("\n--- RESULTADOS PARA O DATASET:", nome_ficheiro, "---")
    print(df_res.to_string(index=False))


def main():
    pasta_data = "data"
    ficheiros = glob.glob(os.path.join(pasta_data, "*.csv"))

    if not ficheiros:
        print(f"Nenhum ficheiro CSV encontrado na pasta '{pasta_data}'")
        return

    print(
        f"1. Ficheiros CSV detetados automaticamente: {[os.path.basename(f) for f in ficheiros]}")

    for f in ficheiros:
        processar_e_treinar_dataset(f)


if __name__ == "__main__":
    main()
