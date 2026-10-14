# src/evaluation.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)


def evaluate_all_models(best_models: dict, X_test, y_test, results_dir="results/"):
    """
    Avalia os modelos nos dados de teste e gera os gráficos e relatórios do Módulo 7[cite: 1].
    """
    os.makedirs(results_dir, exist_ok=True)
    summary_list = []

    # Prepara a figura para desenhar todas as Curvas ROC sobrepostas
    plt.figure(figsize=(10, 7))

    for model_name, model in best_models.items():
        # Previsões de classe
        y_pred = model.predict(X_test)

        # Probabilidades para o cálculo do ROC-AUC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = y_pred

        # 1. Gerar e guardar Matriz de Confusão[cite: 1]
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f"Matriz de Confusão - {model_name}")
        plt.savefig(os.path.join(results_dir, f"cm_{model_name}.png"))
        plt.close()

        # 2. Curva ROC e AUC Score[cite: 1]
        try:
            auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")
        except Exception as e:
            print(f"Não foi possível calcular ROC para {model_name}: {e}")
            auc = 0.0

        # 3. Extrair Precision, Recall e F1-Score do relatório[cite: 1]
        report = classification_report(y_test, y_pred, output_dict=True)
        summary_list.append({
            'Modelo': model_name,
            'Precision': round(report['weighted avg']['precision'], 4),
            'Recall': round(report['weighted avg']['recall'], 4),
            'F1-Score': round(report['weighted avg']['f1-score'], 4),
            'AUC-ROC': round(auc, 4)
        })

    # Finalizar e guardar o gráfico da Curva ROC Comparativa
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.5)')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Comparativo da Curva ROC entre Modelos')
    plt.legend()
    plt.savefig(os.path.join(results_dir, "roc_curve_comparison.png"))
    plt.close()

    # 4. Exportar a Tabela Resumo Comparativa em CSV
    df_results = pd.DataFrame(summary_list)
    df_results.to_csv(os.path.join(
        results_dir, "model_comparison.csv"), index=False)

    print("\n================ Tabela Comparativa de Modelos ================")
    print(df_results.to_string(index=False))

    return df_results
