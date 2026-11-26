"""
Catalogo de modelos clasicos de machine learning y sus rejillas de busqueda.

Cada entrada es un Pipeline de scikit-learn, no un estimador suelto. Esto es
deliberado: el escalado debe ajustarse SOLO con los datos de entrenamiento de
cada pliegue de validacion cruzada. Si se escalara el dataset completo antes
de partirlo, la media y la desviacion usadas incluirian informacion del
conjunto de prueba (fuga de informacion) y las metricas quedarian infladas.

Modelos incluidos (el enunciado exige al menos dos de ML tradicional):

  dummy       Linea base: predice siempre la clase mayoritaria. Cualquier
              modelo debe superarla con holgura o no aporta nada.
  logreg      Regresion logistica multinomial con regularizacion L2.
  knn         K vecinos mas cercanos.
  svm_rbf     Maquina de soporte vectorial con nucleo gaussiano.
  rf          Random Forest (bagging de arboles).
  hgb         Histogram Gradient Boosting, equivalente en scikit-learn de
              XGBoost. Si xgboost esta instalado se agrega tambien 'xgb'.

Todos los modelos sensibles al desbalance usan class_weight='balanced', es
decir el peso de la clase c es

    w_c = N / (K * N_c)

con N el total de muestras, K el numero de clases y N_c el soporte de la
clase c. Equivale a maximizar la exactitud balanceada en lugar de la
exactitud cruda.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.config import RANDOM_STATE

# Un diccionario nombre -> (pipeline, rejilla de hiperparametros).
ModelZoo = dict[str, tuple[Pipeline, dict]]


def get_model_zoo(rapido: bool = False) -> ModelZoo:
    """
    Devuelve el catalogo de modelos con sus rejillas.

    Parameters
    ----------
    rapido : si True usa rejillas reducidas (utiles para depurar el flujo
        completo en menos de un minuto). El informe final debe generarse con
        rapido=False.
    """
    zoo: ModelZoo = {}

    zoo["dummy"] = (
        Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        {},
    )

    zoo["logreg"] = (
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        {"clf__C": [0.1, 1.0, 10.0] if rapido else [0.01, 0.1, 1.0, 10.0, 100.0]},
    )

    zoo["knn"] = (
        Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
        (
            {"clf__n_neighbors": [5, 11], "clf__weights": ["distance"]}
            if rapido
            else {
                "clf__n_neighbors": [3, 5, 7, 11, 15, 21],
                "clf__weights": ["uniform", "distance"],
                # p=1 distancia Manhattan, p=2 distancia euclidiana.
                "clf__p": [1, 2],
            }
        ),
    )

    zoo["svm_rbf"] = (
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        (
            {"clf__C": [1.0, 10.0], "clf__gamma": ["scale"]}
            if rapido
            else {
                "clf__C": [0.1, 1.0, 10.0, 100.0],
                "clf__gamma": ["scale", 0.01, 0.1, 1.0],
            }
        ),
    )

    zoo["rf"] = (
        Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        class_weight="balanced_subsample",
                        # n_jobs=1 a proposito: la paralelizacion la maneja
                        # GridSearchCV. Anidar n_jobs=-1 dentro de una busqueda
                        # tambien paralela multiplica los procesos y agota la
                        # memoria (el equipo de desarrollo tiene 8 GB).
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        (
            {"clf__n_estimators": [300], "clf__max_depth": [None, 12]}
            if rapido
            else {
                "clf__n_estimators": [300, 600],
                "clf__max_depth": [None, 8, 16],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_features": ["sqrt", 0.3],
            }
        ),
    )

    zoo["hgb"] = (
        Pipeline([("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE))]),
        (
            {"clf__max_iter": [200], "clf__learning_rate": [0.1]}
            if rapido
            else {
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_iter": [200, 400],
                "clf__max_leaf_nodes": [15, 31],
                "clf__l2_regularization": [0.0, 1.0],
            }
        ),
    )

    try:  # xgboost es opcional: no esta en todas las maquinas del equipo
        from xgboost import XGBClassifier

        zoo["xgb"] = (
            Pipeline(
                [
                    (
                        "clf",
                        XGBClassifier(
                            tree_method="hist",
                            eval_metric="mlogloss",
                            random_state=RANDOM_STATE,
                            n_jobs=1,  # ver la nota de RandomForest
                        ),
                    )
                ]
            ),
            (
                {"clf__n_estimators": [300], "clf__max_depth": [6]}
                if rapido
                else {
                    "clf__n_estimators": [300, 600],
                    "clf__max_depth": [4, 6, 8],
                    "clf__learning_rate": [0.05, 0.1],
                    "clf__subsample": [0.8, 1.0],
                    "clf__colsample_bytree": [0.8, 1.0],
                }
            ),
        )
    except ModuleNotFoundError:
        pass

    return zoo


def tamano_rejilla(rejilla: dict) -> int:
    """Numero de combinaciones de una rejilla (para estimar el costo)."""
    total = 1
    for valores in rejilla.values():
        total *= len(valores)
    return total
