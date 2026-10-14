# src/models.py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_all_models():
    """
    Retorna um dicionário com todos os 7 algoritmos marcados com (M) no Módulo 6[cite: 1].
    """
    models = {
        # 1. Regressão Logística
        'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),

        # 2. Árvore de Decisão
        'Decision_Tree': DecisionTreeClassifier(random_state=42),

        # 3. k-Nearest Neighbors (k-NN)
        'KNN': KNeighborsClassifier(),

        # 4. Naïve Bayes (Gaussian)
        'Naive_Bayes': GaussianNB(),

        # 5. Support Vector Machines (SVM)
        'SVM': SVC(probability=True, random_state=42),

        # 6. Random Forest (Bagging)
        'Random_Forest': RandomForestClassifier(random_state=42),

        # 7. Gradient Boosting
        'Gradient_Boosting': GradientBoostingClassifier(random_state=42)
    }
    return models


def get_hyperparameter_grids():
    """
    Define os parâmetros de teste para o Hyperparameter Tuning (Módulo 7)[cite: 1].
    """
    param_grids = {
        'Logistic_Regression': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'liblinear']
        },
        'Decision_Tree': {
            'max_depth': [3, 5, 10, None],
            'criterion': ['gini', 'entropy']
        },
        'KNN': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        'Naive_Bayes': {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear']
        },
        'Random_Forest': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None]
        },
        'Gradient_Boosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1]
        }
    }
    return param_grids
