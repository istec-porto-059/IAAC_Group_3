# src/training.py
import os
import joblib
from sklearn.model_selection import GridSearchCV
from src.models import get_all_models, get_hyperparameter_grids


def train_and_tune_all_models(X_train, y_train, save_dir="models/"):
    """
    Treina e otimiza (Hyperparameter Tuning + Cross Validation) todos os modelos[cite: 1].
    Guarda o melhor exemplar de cada algoritmo na pasta 'models/'.
    """
    models = get_all_models()
    param_grids = get_hyperparameter_grids()
    best_models = {}

    os.makedirs(save_dir, exist_ok=True)

    for model_name, model in models.items():
        print(f"\n---> A otimizar e treinar: {model_name}...")
        grid = param_grids.get(model_name, {})

        # O GridSearchCV executa Cross-Validation para encontrar a melhor combinação
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid,
            cv=5,                  # 5-fold cross validation
            # Avalia e otimiza a métrica F1-Score[cite: 1]
            scoring='f1_weighted',
            n_jobs=-1              # Utiliza todos os núcleos do processador
        )

        # Executa o treino
        grid_search.fit(X_train, y_train)

        # Obtém o melhor modelo resultante
        best_model = grid_search.best_estimator_
        best_models[model_name] = best_model

        print(
            f"Melhores parâmetros para {model_name}: {grid_search.best_params_}")

        # Guarda o modelo treinado na pasta 'models/'
        file_path = os.path.join(save_dir, f"{model_name}.pkl")
        joblib.dump(best_model, file_path)

    return best_models
