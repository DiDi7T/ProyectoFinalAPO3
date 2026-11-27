"""
Entrenamiento y ajuste de hiperparametros de los modelos clasicos.

Protocolo experimental
----------------------
1. Las particiones vienen fijas de src.features.build_features y respetan los
   grupos de imagenes casi duplicadas. Aqui NO se vuelven a partir los datos.
2. La busqueda de hiperparametros se hace con GridSearchCV sobre el conjunto
   de ENTRENAMIENTO usando StratifiedGroupKFold de 5 pliegues, pasando los
   identificadores de grupo. Asi ni la validacion cruzada interna sufre fuga
   por duplicados.
3. La metrica de seleccion es F1 macro, no exactitud, porque las clases estan
   desbalanceadas y todas interesan por igual.
4. El conjunto de validacion se usa para comparar modelos entre si; el de
   prueba se evalua una sola vez, al final, con el modelo ya elegido. Ese
   orden es lo que hace que la metrica de prueba sea una estimacion honesta
   del desempeno en produccion.

Uso
---
    python -m src.training.train_classic --task quality
    python -m src.training.train_classic --task both --rapido
    python -m src.training.train_classic --task fruit
    python -m src.training.train_classic --task quality --features baseline
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from src.evaluation import metrics as M
from src.evaluation import plots as P
from src.models.classic import get_model_zoo, tamano_rejilla
from src.utils.config import (
    CHECKPOINTS,
    FEATURES_DIR,
    FRUIT_CLASSES,
    QUALITY_CLASSES,
    RANDOM_STATE,
    RESULTS,
    SIZE_CLASSES,
    ensure_dirs,
    set_global_seed,
)

FIGURAS = RESULTS / "figuras"


# ============================================================
# 1. CARGA DE DATOS CACHEADOS
# ============================================================

def cargar_datos(tipo_features: str = "enriched") -> dict:
    """Carga las matrices cacheadas y arma las mascaras de particion."""
    sufijo = "" if tipo_features == "enriched" else "_baseline"
    ruta_x = FEATURES_DIR / f"X{sufijo}.npy"
    if not ruta_x.exists():
        raise FileNotFoundError(
            f"No existe {ruta_x}. Ejecuta: python -m src.features.build_features "
            f"--features {tipo_features}"
        )

    X = np.load(ruta_x)
    split = np.load(FEATURES_DIR / "split.npy", allow_pickle=True).astype(str)
    index = pd.read_csv(FEATURES_DIR / "index.csv")
    datos = {
        "X": X,
        "y_quality": np.load(FEATURES_DIR / "y_quality.npy"),
        "y_size": np.load(FEATURES_DIR / "y_size.npy"),
        "y_fruit": (
            np.load(FEATURES_DIR / "y_fruit.npy")
            if (FEATURES_DIR / "y_fruit.npy").exists()
            else None
        ),
        "grupo": np.load(FEATURES_DIR / "grupo.npy"),
        "split": split,
        "train": split == "train",
        "val": split == "val",
        "test": split == "test",
        "index": index,
        "source": (
            index["source"].astype(str).to_numpy()
            if "source" in index.columns
            else np.full(len(index), "desconocida")
        ),
    }

    meta_path = FEATURES_DIR / f"meta{sufijo}.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            datos["meta"] = json.load(fh)
    return datos


# ============================================================
# 2. ENTRENAMIENTO DE UN MODELO
# ============================================================

def entrenar_modelo(
    nombre: str,
    pipeline,
    rejilla: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    grupos_train: np.ndarray,
    n_pliegues: int = 5,
    n_jobs: int = 2,
) -> dict:
    """Ajusta un modelo con GridSearchCV respetando los grupos de duplicados."""
    n_clases = len(np.unique(y_train))
    if n_clases < 2:
        raise ValueError(f"El conjunto de entrenamiento tiene {n_clases} clase(s)")

    cv = StratifiedGroupKFold(n_splits=n_pliegues, shuffle=True, random_state=RANDOM_STATE)

    busqueda = GridSearchCV(
        pipeline,
        rejilla,
        scoring="f1_macro",
        cv=cv,
        # n_jobs moderado y pre_dispatch acotado: cada proceso recibe una copia
        # de la matriz de caracteristicas, de modo que n_jobs=-1 en una maquina
        # de 8 GB agota la memoria y el proceso muere sin traza.
        n_jobs=n_jobs,
        pre_dispatch="2*n_jobs",
        refit=True,
        return_train_score=True,
        error_score="raise",
    )

    t0 = time.perf_counter()
    busqueda.fit(X_train, y_train, groups=grupos_train)
    segundos = time.perf_counter() - t0

    idx = busqueda.best_index_
    return {
        "nombre": nombre,
        "estimador": busqueda.best_estimator_,
        "mejores_params": {k: str(v) for k, v in busqueda.best_params_.items()},
        "cv_f1_macro": float(busqueda.best_score_),
        "cv_f1_macro_std": float(busqueda.cv_results_["std_test_score"][idx]),
        "cv_f1_macro_train": float(busqueda.cv_results_["mean_train_score"][idx]),
        "combinaciones": int(max(tamano_rejilla(rejilla), 1)),
        "segundos_busqueda": round(segundos, 1),
    }


def _probabilidades(estimador, X: np.ndarray) -> np.ndarray | None:
    if hasattr(estimador, "predict_proba"):
        try:
            return estimador.predict_proba(X)
        except (AttributeError, ValueError):
            return None
    return None


def _importancias(estimador, n_features: int) -> np.ndarray | None:
    """Importancia de caracteristicas si el modelo la expone."""
    clf = estimador[-1] if hasattr(estimador, "__getitem__") else estimador
    if hasattr(clf, "feature_importances_"):
        return np.asarray(clf.feature_importances_, dtype=float)
    if hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_, dtype=float)
        return np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    return None


# ============================================================
# 3. EXPERIMENTO COMPLETO DE UNA TAREA
# ============================================================

def ejecutar_tarea(
    tarea: str,
    datos: dict,
    rapido: bool = False,
    modelos: list[str] | None = None,
    tipo_features: str = "enriched",
    fuente: str = "all",
    etiqueta: str = "",
    n_jobs: int = 2,
) -> dict:
    """
    Entrena todo el catalogo para una tarea ('quality' o 'size') y escribe
    metricas, figuras y modelos entrenados.

    Parameters
    ----------
    fuente : 'all' usa todo el dataset; 'kaggle' o 'recolectado' restringen el
        experimento a una sola fuente. Restringir a 'kaggle' convierte el
        problema en binario (ese dataset no tiene clase 'regular'), que es la
        configuracion comparable con los trabajos publicados sobre el.
    etiqueta : sufijo para los archivos de salida, de modo que varios
        experimentos coexistan sin sobreescribirse (p. ej. 'kaggle2clases').
    """
    ensure_dirs()
    FIGURAS.mkdir(parents=True, exist_ok=True)

    # 'fruit' identifica el tipo de fruta. No es una salida del enunciado, pero
    # la interfaz la usa para decir "esta manzana tiene calidad ...".
    if tarea == "fruit" and datos.get("y_fruit") is None:
        raise FileNotFoundError(
            "No existe y_fruit.npy. Ejecuta: python -m src.features.build_features"
        )
    y = {"quality": datos["y_quality"], "size": datos["y_size"], "fruit": datos.get("y_fruit")}[tarea]
    nombres_clases = {"quality": QUALITY_CLASSES, "size": SIZE_CLASSES, "fruit": FRUIT_CLASSES}[tarea]
    es_ordinal = tarea == "size"

    tr, va, te = datos["train"], datos["val"], datos["test"]
    X = datos["X"]

    if fuente != "all":
        de_la_fuente = datos["source"] == fuente
        if de_la_fuente.sum() == 0:
            raise ValueError(f"No hay imagenes de la fuente '{fuente}' en el cache")
        # Se filtra DESPUES de la particion, nunca antes: las particiones son
        # las mismas para todos los experimentos y asi siguen siendo comparables.
        tr, va, te = tr & de_la_fuente, va & de_la_fuente, te & de_la_fuente

    sufijo_salida = f"_{etiqueta}" if etiqueta else ""
    nombre_tarea = f"{tarea}{sufijo_salida}"

    print("=" * 70)
    print(f"TAREA: {tarea}   caracteristicas: {tipo_features}   X: {X.shape}")
    print(f"fuente: {fuente}   etiqueta de salida: {etiqueta or '(ninguna)'}")
    print(f"train={tr.sum()}  val={va.sum()}  test={te.sum()}")
    presentes, nombres_presentes = M.clases_presentes(y, nombres_clases)
    print(f"clases presentes: {nombres_presentes} de {list(nombres_clases)}")
    if len(presentes) < len(nombres_clases):
        print(
            "[AVISO] Faltan clases del espacio de etiquetas definido. Las metricas "
            "solo son comparables entre modelos, no con el sistema final de 3 clases."
        )
    print("=" * 70)

    zoo = get_model_zoo(rapido=rapido)
    if modelos:
        zoo = {k: v for k, v in zoo.items() if k in modelos}

    resultados_val: dict[str, dict] = {}
    resultados_test: dict[str, dict] = {}
    detalles: dict[str, dict] = {}
    predicciones_test: dict[str, np.ndarray] = {}

    for nombre, (pipeline, rejilla) in zoo.items():
        print(f"\n--- {nombre} ({max(tamano_rejilla(rejilla), 1)} combinaciones) ---")
        ajuste = entrenar_modelo(
            nombre, pipeline, rejilla, X[tr], y[tr], datos["grupo"][tr], n_jobs=n_jobs
        )
        estimador = ajuste.pop("estimador")

        pred_val = estimador.predict(X[va])
        pred_test = estimador.predict(X[te])
        predicciones_test[nombre] = pred_test

        m_val = M.calcular_metricas(
            y[va], pred_val, nombres_clases, _probabilidades(estimador, X[va]), ordinal=es_ordinal
        )
        m_test = M.calcular_metricas(
            y[te], pred_test, nombres_clases, _probabilidades(estimador, X[te]), ordinal=es_ordinal
        )
        if tarea == "quality" and len(presentes) >= 2:
            # idx_bad es la ultima clase del espacio de etiquetas ('bad').
            m_test["costo"] = M.costo_asimetrico(
                y[te], pred_test, idx_good=0, idx_bad=len(nombres_clases) - 1
            )

        resultados_val[nombre] = m_val
        resultados_test[nombre] = m_test
        detalles[nombre] = ajuste

        print(
            f"  CV  F1 macro = {ajuste['cv_f1_macro']:.4f} "
            f"(+/- {ajuste['cv_f1_macro_std']:.4f})  "
            f"train = {ajuste['cv_f1_macro_train']:.4f}"
        )
        print(f"  VAL F1 macro = {m_val['f1_macro']:.4f}  acc = {m_val['accuracy']:.4f}")
        print(f"  TEST F1 macro = {m_test['f1_macro']:.4f}  acc = {m_test['accuracy']:.4f}")
        print(f"  mejores params: {ajuste['mejores_params']}")

        joblib.dump(estimador, CHECKPOINTS / f"classic_{nombre_tarea}_{nombre}.joblib")

        # Figuras por modelo: matriz de confusion en prueba e importancias.
        P.matriz_confusion(
            m_test["matriz_confusion"],
            m_test["etiquetas_matriz"],
            f"{nombre} - {tarea} (prueba)",
            FIGURAS / f"cm_{nombre_tarea}_{nombre}",
        )
        if nombre != "dummy":
            imp = _importancias(estimador, X.shape[1])
            nombres_feat = (datos.get("meta", {}) or {}).get("nombres_features") or [
                f"f{i}" for i in range(X.shape[1])
            ]
            if imp is not None and len(imp) == len(nombres_feat):
                P.importancia_caracteristicas(
                    nombres_feat,
                    imp,
                    f"Importancia de caracteristicas - {nombre} ({tarea})",
                    FIGURAS / f"importancia_{nombre_tarea}_{nombre}",
                )

    # ---------- comparativas y figuras globales ----------
    tabla_val = M.tabla_comparativa(resultados_val)
    tabla_test = M.tabla_comparativa(resultados_test)

    print("\n=== VALIDACION (seleccion de modelo) ===")
    print(tabla_val.to_string(index=False))
    print("\n=== PRUEBA ===")
    print(tabla_test.to_string(index=False))

    tabla_val.to_csv(RESULTS / f"classic_{nombre_tarea}_val_leaderboard.csv", index=False)
    tabla_test.to_csv(RESULTS / f"classic_{nombre_tarea}_test_leaderboard.csv", index=False)

    candidatos = [n for n in resultados_val if n != "dummy"]
    mejor = max(candidatos, key=lambda n: resultados_val[n]["f1_macro"]) if candidatos else None

    if mejor:
        P.comparacion_modelos(
            list(resultados_val.keys()),
            [resultados_val[n]["f1_macro"] for n in resultados_val],
            "F1 macro (validacion)",
            FIGURAS / f"comparacion_{nombre_tarea}",
            referencia=resultados_val.get("dummy", {}).get("f1_macro"),
        )
        estimador_mejor = joblib.load(CHECKPOINTS / f"classic_{nombre_tarea}_{mejor}.joblib")
        proba_test = _probabilidades(estimador_mejor, X[te])
        if proba_test is not None:
            P.curvas_roc(
                y[te],
                proba_test,
                resultados_test[mejor]["etiquetas_matriz"],
                f"Curvas ROC - {mejor} ({tarea}, prueba)",
                FIGURAS / f"roc_{nombre_tarea}_{mejor}",
            )
        joblib.dump(estimador_mejor, CHECKPOINTS / f"mejor_classic_{nombre_tarea}.joblib")
        print(f"\nModelo seleccionado por validacion: {mejor}")
        print(f"  -> {CHECKPOINTS / f'mejor_classic_{nombre_tarea}.joblib'}")

    informe = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "tarea": tarea,
        "tipo_features": tipo_features,
        "rapido": rapido,
        "n_features": int(X.shape[1]),
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "n_test": int(te.sum()),
        "fuente": fuente,
        "etiqueta": etiqueta,
        "clases_definidas": list(nombres_clases),
        "clases_presentes": nombres_presentes,
        "modelo_seleccionado": mejor,
        "ajuste": detalles,
        "metricas_validacion": resultados_val,
        "metricas_prueba": resultados_test,
    }
    destino = RESULTS / f"classic_{nombre_tarea}_metrics.json"
    with open(destino, "w", encoding="utf-8") as fh:
        json.dump(informe, fh, indent=2, ensure_ascii=False)
    print(f"Metricas completas: {destino}")

    return informe


def main() -> None:
    from src.utils.registro import registrar_ejecucion

    with registrar_ejecucion("train_classic"):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--task", choices=("quality", "size", "fruit", "both", "all"), default="quality"
        )
        parser.add_argument("--features", choices=("enriched", "baseline"), default="enriched")
        parser.add_argument("--rapido", action="store_true", help="Rejillas reducidas")
        parser.add_argument("--models", nargs="*", default=None, help="Subconjunto de modelos")
        parser.add_argument(
            "--source",
            choices=("all", "kaggle", "recolectado"),
            default="all",
            help="Restringir el experimento a una fuente de datos",
        )
        parser.add_argument(
            "--jobs",
            type=int,
            default=2,
            help="Procesos paralelos de GridSearchCV (bajar si la memoria es escasa)",
        )
        parser.add_argument(
            "--tag",
            default="",
            help="Sufijo para los archivos de salida, para no sobreescribir otro experimento",
        )
        args = parser.parse_args()

        set_global_seed()
        datos = cargar_datos(args.features)

        tareas = {"both": ("quality", "size"), "all": ("quality", "size", "fruit")}.get(
            args.task, (args.task,)
        )
        for tarea in tareas:
            ejecutar_tarea(
                tarea,
                datos,
                rapido=args.rapido,
                modelos=args.models,
                tipo_features=args.features,
                fuente=args.source,
                etiqueta=args.tag,
                n_jobs=args.jobs,
            )


if __name__ == "__main__":
    main()
