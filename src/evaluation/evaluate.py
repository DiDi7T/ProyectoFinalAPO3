"""
Evaluacion consolidada: reune todos los experimentos en un solo informe.

Este modulo no entrena nada. Lee los artefactos que dejaron
src.training.train_classic y src.training.train_cnn y produce el material que
va directamente al informe final:

1. Tabla comparativa unica de todas las familias de modelos (linea base, ML
   clasico y CNN) sobre el mismo conjunto de prueba.
2. Analisis de generalizacion entre dominios: el mismo modelo evaluado por
   separado en las imagenes de Kaggle (estudio, fondo uniforme) y en las
   fotografias propias de plaza de mercado. Es la pregunta que de verdad
   importa para el despliegue: un modelo que acierta 97 % en Kaggle y 70 % en
   fotos reales no sirve en una bodega.
3. Desglose de errores por fruta y por clase, para distinguir un fallo del
   modelo de un fallo de la etiqueta o de la segmentacion.
4. Brecha de generalizacion (train - test) de cada modelo, que es la evidencia
   cuantitativa de sobreajuste que pide la rubrica.

Uso
---
    python -m src.evaluation.evaluate
    python -m src.evaluation.evaluate --task quality
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.evaluation import plots as P
from src.utils.config import (
    CHECKPOINTS,
    FEATURES_DIR,
    QUALITY_CLASSES,
    RESULTS,
    SIZE_CLASSES,
    ensure_dirs,
)

FIGURAS = RESULTS / "figuras"


# ============================================================
# 1. CARGA DE ARTEFACTOS
# ============================================================

def _leer_json(ruta):
    if not ruta.exists():
        return None
    with open(ruta, encoding="utf-8") as fh:
        return json.load(fh)


def cargar_contexto() -> dict:
    """Carga features, etiquetas, particiones y el indice con la fuente."""
    if not (FEATURES_DIR / "X.npy").exists():
        raise FileNotFoundError(
            "Faltan las features. Ejecuta: python -m src.features.build_features"
        )

    split = np.load(FEATURES_DIR / "split.npy", allow_pickle=True).astype(str)
    index = pd.read_csv(FEATURES_DIR / "index.csv")
    return {
        "X": np.load(FEATURES_DIR / "X.npy"),
        "y_quality": np.load(FEATURES_DIR / "y_quality.npy"),
        "y_size": np.load(FEATURES_DIR / "y_size.npy"),
        "split": split,
        "test": split == "test",
        "index": index,
        "source": (
            index["source"].astype(str).to_numpy()
            if "source" in index.columns
            else np.full(len(index), "desconocida")
        ),
        "fruit": index["fruit"].astype(str).to_numpy(),
    }


# ============================================================
# 2. TABLA COMPARATIVA GLOBAL
# ============================================================

def tabla_global(tarea: str) -> pd.DataFrame:
    """Une el leaderboard de los modelos clasicos con el resultado de la CNN."""
    filas: list[dict] = []

    informe_clasico = _leer_json(RESULTS / f"classic_{tarea}_metrics.json")
    if informe_clasico:
        validacion = informe_clasico.get("metricas_validacion", {})
        for nombre, m in informe_clasico["metricas_prueba"].items():
            ajuste = informe_clasico.get("ajuste", {}).get(nombre, {})
            filas.append(
                {
                    "familia": "linea base" if nombre == "dummy" else "ML clasico",
                    "modelo": nombre,
                    "accuracy": m.get("accuracy"),
                    "balanced_accuracy": m.get("balanced_accuracy"),
                    "f1_macro": m.get("f1_macro"),
                    "cohen_kappa": m.get("cohen_kappa"),
                    "mcc": m.get("mcc"),
                    "roc_auc_ovr": m.get("roc_auc_ovr"),
                    "mae_ordinal": m.get("mae_ordinal"),
                    "val_f1_macro": (validacion.get(nombre) or {}).get("f1_macro"),
                    "cv_f1_macro": ajuste.get("cv_f1_macro"),
                    "cv_f1_macro_train": ajuste.get("cv_f1_macro_train"),
                    "falsos_buenos": (m.get("costo") or {}).get("n_falsos_buenos"),
                }
            )

    informe_cnn = _leer_json(RESULTS / "cnn_metrics.json")
    if informe_cnn:
        clave = "metricas_prueba_calidad" if tarea == "quality" else "metricas_prueba_tamano"
        m = informe_cnn.get(clave, {})
        hist = informe_cnn.get("historial", {})
        mejor = informe_cnn.get("mejor_epoca", 1)
        pos = mejor - 1 if 0 <= mejor - 1 < len(hist.get("f1_quality_train", [])) else -1
        filas.append(
            {
                "familia": "deep learning",
                "modelo": "cnn",
                "accuracy": m.get("accuracy"),
                "balanced_accuracy": m.get("balanced_accuracy"),
                "f1_macro": m.get("f1_macro"),
                "cohen_kappa": m.get("cohen_kappa"),
                "mcc": m.get("mcc"),
                "roc_auc_ovr": m.get("roc_auc_ovr"),
                "mae_ordinal": m.get("mae_ordinal"),
                # La CNN no pasa por validacion cruzada: su hiperparametrizacion
                # se controla con la particion de validacion y parada temprana.
                # Dejar su F1 de validacion en la columna 'cv_f1_macro' la haria
                # comparable con el criterio de seleccion de los modelos
                # clasicos, que es una media de 5 pliegues, y no lo es. Va en
                # 'val_f1_macro', igual que la de los clasicos.
                "val_f1_macro": (hist.get("f1_quality_val") or [None])[pos]
                if tarea == "quality"
                else None,
                "cv_f1_macro": None,
                "cv_f1_macro_train": (hist.get("f1_quality_train") or [None])[pos]
                if tarea == "quality"
                else None,
                "falsos_buenos": (m.get("costo") or {}).get("n_falsos_buenos"),
            }
        )

    informe_cnn_mask = _leer_json(RESULTS / "cnn_metrics_mascara.json")
    if informe_cnn_mask and tarea == "quality":
        m = informe_cnn_mask.get("metricas_prueba_calidad", {})
        filas.append(
            {
                "familia": "deep learning",
                "modelo": "cnn (sin fondo)",
                "accuracy": m.get("accuracy"),
                "balanced_accuracy": m.get("balanced_accuracy"),
                "f1_macro": m.get("f1_macro"),
                "cohen_kappa": m.get("cohen_kappa"),
                "mcc": m.get("mcc"),
                "roc_auc_ovr": m.get("roc_auc_ovr"),
                "falsos_buenos": (m.get("costo") or {}).get("n_falsos_buenos"),
            }
        )

    if not filas:
        raise FileNotFoundError(
            "No hay resultados que consolidar. Entrena primero con "
            "src.training.train_classic y src.training.train_cnn."
        )

    df = pd.DataFrame(filas)
    # Brecha de generalizacion: cuanto cae el desempeno al salir del
    # entrenamiento. Valores altos indican memorizacion.
    df["brecha_train_test"] = df["cv_f1_macro_train"] - df["f1_macro"]
    return df.sort_values("f1_macro", ascending=False).reset_index(drop=True)


# ============================================================
# 3. GENERALIZACION ENTRE DOMINIOS
# ============================================================

def analisis_por_fuente(tarea: str, ctx: dict) -> pd.DataFrame | None:
    """
    Evalua el mejor modelo clasico por separado en cada fuente de datos.

    Se reutiliza el modelo ya entrenado (mejor_classic_<tarea>.joblib) y solo
    se cambia el subconjunto de prueba. Ninguna metrica aqui influye en la
    seleccion del modelo: es un analisis posterior.
    """
    ruta = CHECKPOINTS / f"mejor_classic_{tarea}.joblib"
    if not ruta.exists():
        print(f"[AVISO] No existe {ruta}; se omite el analisis por fuente.")
        return None

    modelo = joblib.load(ruta)
    nombres = QUALITY_CLASSES if tarea == "quality" else SIZE_CLASSES
    y = ctx["y_quality"] if tarea == "quality" else ctx["y_size"]
    te = ctx["test"]

    filas = []
    for fuente in sorted(set(ctx["source"][te])):
        sel = te & (ctx["source"] == fuente)
        if sel.sum() < 5:
            continue
        pred = modelo.predict(ctx["X"][sel])
        m = M.calcular_metricas(y[sel], pred, nombres, ordinal=(tarea == "size"))
        filas.append(
            {
                "fuente": fuente,
                "n": int(sel.sum()),
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "f1_macro": m["f1_macro"],
                "clases_presentes": ", ".join(m["clases_presentes"]),
            }
        )

    # Y el conjunto completo, como referencia.
    pred_total = modelo.predict(ctx["X"][te])
    m_total = M.calcular_metricas(y[te], pred_total, nombres, ordinal=(tarea == "size"))
    filas.append(
        {
            "fuente": "TODAS",
            "n": int(te.sum()),
            "accuracy": m_total["accuracy"],
            "balanced_accuracy": m_total["balanced_accuracy"],
            "f1_macro": m_total["f1_macro"],
            "clases_presentes": ", ".join(m_total["clases_presentes"]),
        }
    )
    return pd.DataFrame(filas)


def analisis_por_fruta(tarea: str, ctx: dict) -> pd.DataFrame | None:
    """Desempeno del mejor modelo clasico desglosado por fruta."""
    ruta = CHECKPOINTS / f"mejor_classic_{tarea}.joblib"
    if not ruta.exists():
        return None

    modelo = joblib.load(ruta)
    nombres = QUALITY_CLASSES if tarea == "quality" else SIZE_CLASSES
    y = ctx["y_quality"] if tarea == "quality" else ctx["y_size"]
    te = ctx["test"]
    pred_total = modelo.predict(ctx["X"][te])

    filas = []
    frutas_test = ctx["fruit"][te]
    y_test = y[te]
    for fruta in sorted(set(frutas_test)):
        sel = frutas_test == fruta
        if sel.sum() < 5:
            continue
        m = M.calcular_metricas(y_test[sel], pred_total[sel], nombres, ordinal=(tarea == "size"))
        filas.append(
            {
                "fruta": fruta,
                "n": int(sel.sum()),
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "errores": int((y_test[sel] != pred_total[sel]).sum()),
            }
        )
    return pd.DataFrame(filas).sort_values("f1_macro").reset_index(drop=True)


# ============================================================
# 4. INFORME EN MARKDOWN
# ============================================================

def _md(df: pd.DataFrame, decimales: int = 4) -> str:
    return df.round(decimales).to_markdown(index=False)


def generar_informe(tareas: tuple[str, ...] = ("quality", "size")) -> str:
    """Escribe experiments/results/informe_consolidado.md y lo devuelve."""
    ensure_dirs()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    ctx = cargar_contexto()

    meta = _leer_json(FEATURES_DIR / "meta.json") or {}
    partes: list[str] = [
        "# Informe consolidado de resultados",
        "",
        f"Generado automaticamente el {datetime.now().strftime('%Y-%m-%d %H:%M')} "
        "por `python -m src.evaluation.evaluate`.",
        "",
        "## Conjunto de datos",
        "",
        f"- Imagenes totales: **{meta.get('n_imagenes', len(ctx['index']))}**",
        f"- Caracteristicas por imagen: **{meta.get('n_features', ctx['X'].shape[1])}**",
        f"- Grupos de imagenes casi duplicadas: **{meta.get('n_grupos_duplicados', 'n/d')}**",
        f"- Particiones: {meta.get('conteo_split', {})}",
        f"- Clases de calidad: {meta.get('conteo_quality', {})}",
        f"- Clases de tamano: {meta.get('conteo_size', {})}",
        "",
    ]

    if "source" in ctx["index"].columns:
        cruce = pd.crosstab(
            ctx["index"]["source"], ctx["index"]["quality"], margins=True, margins_name="TOTAL"
        )
        partes += ["### Composicion por fuente y calidad", "", cruce.to_markdown(), ""]
        cruce_split = pd.crosstab(
            ctx["index"]["split"], ctx["index"]["source"], margins=True, margins_name="TOTAL"
        )
        partes += ["### Fuente por particion", "", cruce_split.to_markdown(), ""]

    for tarea in tareas:
        titulo = "calidad" if tarea == "quality" else "tamano"
        try:
            global_df = tabla_global(tarea)
        except FileNotFoundError as exc:
            partes += [f"## Tarea: {titulo}", "", f"_Sin resultados: {exc}_", ""]
            continue

        partes += [
            f"## Tarea: {titulo}",
            "",
            "### Comparacion de todos los modelos en el conjunto de prueba",
            "",
            _md(global_df.drop(columns=["mae_ordinal"] if tarea == "quality" else [])),
            "",
        ]

        global_df.to_csv(RESULTS / f"consolidado_{tarea}.csv", index=False)

        por_fuente = analisis_por_fuente(tarea, ctx)
        if por_fuente is not None and len(por_fuente):
            partes += [
                "### Generalizacion entre dominios (mejor modelo clasico)",
                "",
                _md(por_fuente),
                "",
                "La diferencia entre las filas `kaggle` y `recolectado` mide el cambio de "
                "dominio: las imagenes publicas estan tomadas en condiciones controladas, "
                "mientras que las propias tienen fondos, sombras e iluminacion reales.",
                "",
            ]
            por_fuente.to_csv(RESULTS / f"por_fuente_{tarea}.csv", index=False)

        por_fruta = analisis_por_fruta(tarea, ctx)
        if por_fruta is not None and len(por_fruta):
            partes += [
                "### Desempeno por fruta (mejor modelo clasico)",
                "",
                _md(por_fruta),
                "",
            ]
            por_fruta.to_csv(RESULTS / f"por_fruta_{tarea}.csv", index=False)

        # Figura comparativa de todas las familias.
        validos = global_df.dropna(subset=["f1_macro"])
        if len(validos):
            referencia = validos.loc[validos["modelo"] == "dummy", "f1_macro"]
            P.comparacion_modelos(
                validos["modelo"].tolist(),
                validos["f1_macro"].tolist(),
                f"F1 macro - {titulo} (prueba)",
                FIGURAS / f"consolidado_{tarea}",
                referencia=float(referencia.iloc[0]) if len(referencia) else None,
            )

    partes += [
        "## Como leer estos numeros",
        "",
        "- **f1_macro** es la metrica de seleccion: promedia el F1 de cada clase sin "
        "ponderar por soporte, asi que una clase minoritaria pesa igual que una mayoritaria.",
        "- **brecha_train_test** es la diferencia entre el F1 de entrenamiento y el de "
        "prueba. Valores por encima de ~0,05 indican sobreajuste.",
        "- **falsos_buenos** cuenta las frutas realmente malas clasificadas como buenas. "
        "Es el error costoso del negocio y conviene minimizarlo incluso sacrificando "
        "algo de exactitud global.",
        "- Las particiones respetan grupos de imagenes casi duplicadas, por lo que estas "
        "cifras no estan infladas por fuga de informacion.",
        "",
    ]

    texto = "\n".join(partes)
    destino = RESULTS / "informe_consolidado.md"
    destino.write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\nInforme escrito en: {destino}")
    return texto


def main() -> None:
    from src.utils.registro import registrar_ejecucion

    with registrar_ejecucion("evaluate"):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--task", choices=("quality", "size", "both"), default="both")
        args = parser.parse_args()

        tareas = ("quality", "size") if args.task == "both" else (args.task,)
        generar_informe(tareas)


if __name__ == "__main__":
    main()
