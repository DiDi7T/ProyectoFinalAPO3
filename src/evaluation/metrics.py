"""
Metricas de desempeno del sistema de clasificacion de calidad.

Por que no basta la exactitud
-----------------------------
La exactitud (accuracy) es enganosa en este problema por dos razones:

1. Las clases estan desbalanceadas. Con 'regular' subrepresentada, un modelo
   que nunca prediga esa clase puede superar el 85 % de exactitud y ser
   inservible justo en la categoria que decide si la fruta se vende con
   descuento o se descarta.
2. Los errores no cuestan lo mismo. Clasificar una fruta mala como buena
   (falso 'good') envia producto deteriorado al cliente; el error inverso
   solo genera una perdida de margen. Por eso se reporta la matriz de
   confusion completa y el recall por clase, no solo un promedio.

Metricas reportadas
-------------------
accuracy             Proporcion de aciertos.
balanced_accuracy    Media de los recalls por clase; insensible al desbalance.
f1_macro             Media no ponderada de los F1 por clase.
f1_weighted          F1 ponderado por soporte.
precision/recall     Macro y por clase.
cohen_kappa          Acuerdo corregido por azar.
mcc                  Coeficiente de correlacion de Matthews.
roc_auc_ovr          AUC macro una-contra-el-resto (si hay probabilidades).
mae_ordinal          Solo para tamano: error absoluto medio sobre los indices
                     ordinales (pequeno=0, mediano=1, grande=2). Penaliza mas
                     confundir pequeno con grande que pequeno con mediano,
                     algo que ninguna metrica nominal captura.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def clases_presentes(y: np.ndarray, nombres: tuple[str, ...]) -> tuple[list[int], list[str]]:
    """
    Indices y nombres de las clases realmente presentes en y.

    Es necesario porque el espacio de etiquetas del proyecto tiene 3 clases de
    calidad, pero mientras no se termine la anotacion manual de la clase
    'regular' solo hay 2 presentes. Sin este filtro scikit-learn reporta
    metricas 0.0 para la clase ausente y contamina los promedios macro.
    """
    presentes = sorted({int(v) for v in np.unique(y)})
    return presentes, [nombres[i] for i in presentes]


def calcular_metricas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nombres_clases: tuple[str, ...],
    y_proba: np.ndarray | None = None,
    ordinal: bool = False,
) -> dict:
    """
    Calcula el diccionario completo de metricas de una prediccion.

    Parameters
    ----------
    y_true, y_pred : vectores de etiquetas enteras.
    nombres_clases : nombres en el orden del espacio de etiquetas global.
    y_proba : matriz (n, n_clases_presentes) de probabilidades, opcional.
    ordinal : True para la tarea de tamano, que anade mae_ordinal.
    """
    etiquetas, nombres = clases_presentes(np.concatenate([y_true, y_pred]), nombres_clases)

    resultado: dict = {
        "n_muestras": int(len(y_true)),
        "clases_presentes": nombres,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=etiquetas).tolist(),
        "etiquetas_matriz": nombres,
        "reporte_por_clase": classification_report(
            y_true,
            y_pred,
            labels=etiquetas,
            target_names=nombres,
            output_dict=True,
            zero_division=0,
        ),
    }

    if ordinal:
        resultado["mae_ordinal"] = float(np.mean(np.abs(y_true.astype(int) - y_pred.astype(int))))

    if y_proba is not None and y_proba.ndim == 2 and len(etiquetas) > 1:
        try:
            if y_proba.shape[1] == 2:
                resultado["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                resultado["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except ValueError as exc:  # p. ej. una sola clase en y_true
            resultado["roc_auc_ovr"] = None
            resultado["roc_auc_error"] = str(exc)

    return resultado


def costo_asimetrico(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_good: int = 0,
    idx_bad: int = 2,
    costo_falso_bueno: float = 5.0,
    costo_falso_malo: float = 1.0,
) -> dict:
    """
    Riesgo empirico con costos asimetricos, pensado para el negocio.

    Se define la matriz de costos

        C[i][j] = costo de predecir j cuando la verdad es i,

    con C = 0 en la diagonal, costo_falso_bueno cuando una fruta mala se
    predice buena y costo_falso_malo en el resto de errores. El riesgo
    empirico es

        R = (1/N) sum_n C[y_n][yhat_n]

    Reportar R junto con F1 permite argumentar la seleccion del modelo en
    terminos del negocio y no solo estadisticos: un modelo con F1 algo menor
    pero menos falsos 'good' puede ser el preferible.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    costos = np.where(
        y_true == y_pred,
        0.0,
        np.where((y_true == idx_bad) & (y_pred == idx_good), costo_falso_bueno, costo_falso_malo),
    )
    return {
        "riesgo_empirico": float(costos.mean()),
        "n_falsos_buenos": int(((y_true == idx_bad) & (y_pred == idx_good)).sum()),
        "n_falsos_malos": int(((y_true == idx_good) & (y_pred == idx_bad)).sum()),
        "costo_falso_bueno": costo_falso_bueno,
        "costo_falso_malo": costo_falso_malo,
    }


def tabla_comparativa(resultados: dict[str, dict], claves: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "cohen_kappa",
    "mcc",
    "roc_auc_ovr",
)) -> "object":
    """Devuelve un DataFrame ordenado por f1_macro para el informe final."""
    import pandas as pd

    filas = []
    for nombre, m in resultados.items():
        fila = {"modelo": nombre}
        fila.update({k: m.get(k) for k in claves})
        filas.append(fila)
    df = pd.DataFrame(filas)
    if "f1_macro" in df.columns:
        df = df.sort_values("f1_macro", ascending=False)
    return df.reset_index(drop=True)
