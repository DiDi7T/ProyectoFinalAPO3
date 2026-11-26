"""
Figuras del informe final.

La rubrica exige graficos de calidad vectorial, por lo que cada funcion
guarda la figura dos veces: en SVG (vectorial, el que va al informe en LaTeX
o Word) y en PNG a 200 ppp (comodo para revisar en el repositorio y en la
presentacion). Todas las funciones devuelven la ruta del SVG.

No se usa seaborn a proposito: matplotlib basta y evita una dependencia mas
en las maquinas del equipo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: necesario para ejecutar por consola

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",  # texto editable y seleccionable en el SVG
    }
)


def _guardar(fig: "plt.Figure", destino: Path) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    svg = destino.with_suffix(".svg")
    fig.savefig(svg, format="svg")
    fig.savefig(destino.with_suffix(".png"), format="png", dpi=200)
    plt.close(fig)
    return svg


def matriz_confusion(
    cm: np.ndarray | list,
    etiquetas: list[str],
    titulo: str,
    destino: Path,
    normalizar: bool = True,
) -> Path:
    """
    Matriz de confusion con anotaciones de conteo y porcentaje por fila.

    La normalizacion por fila (recall por clase) es la lectura correcta para
    este problema: responde '<de las frutas realmente malas, que fraccion
    detecto el modelo?>', que es la pregunta del negocio.
    """
    cm = np.asarray(cm, dtype=float)
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
    matriz = cm_norm if normalizar else cm

    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(etiquetas), 1.4 + 1.0 * len(etiquetas)))
    im = ax.imshow(matriz, cmap="Blues", vmin=0, vmax=matriz.max() if matriz.max() > 0 else 1)
    ax.grid(False)

    ax.set_xticks(range(len(etiquetas)), etiquetas, rotation=30, ha="right")
    ax.set_yticks(range(len(etiquetas)), etiquetas)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(titulo)

    umbral = matriz.max() / 2 if matriz.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            texto = f"{int(cm[i, j])}\n{cm_norm[i, j]:.0%}"
            ax.text(
                j,
                i,
                texto,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if matriz[i, j] > umbral else "black",
            )

    fig.colorbar(im, ax=ax, shrink=0.8, label="Proporcion por fila" if normalizar else "Conteo")
    return _guardar(fig, destino)


def curvas_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    etiquetas: list[str],
    titulo: str,
    destino: Path,
) -> Path:
    """Curvas ROC una-contra-el-resto con su AUC."""
    from sklearn.metrics import auc, roc_curve

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    clases = sorted({int(v) for v in np.unique(y_true)})

    for pos, c in enumerate(clases):
        if y_proba.shape[1] == 2:
            puntajes = y_proba[:, 1] if pos == 1 else 1 - y_proba[:, 1]
        else:
            puntajes = y_proba[:, pos]
        fpr, tpr, _ = roc_curve((y_true == c).astype(int), puntajes)
        ax.plot(fpr, tpr, lw=1.8, label=f"{etiquetas[pos]} (AUC = {auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(titulo)
    ax.legend(loc="lower right", fontsize=8)
    return _guardar(fig, destino)


def importancia_caracteristicas(
    nombres: list[str],
    importancias: np.ndarray,
    titulo: str,
    destino: Path,
    top: int = 20,
) -> Path:
    """Barras horizontales con las 'top' caracteristicas mas influyentes."""
    importancias = np.asarray(importancias, dtype=float)
    orden = np.argsort(importancias)[::-1][:top][::-1]

    fig, ax = plt.subplots(figsize=(6.2, 0.28 * len(orden) + 1.2))
    ax.barh([nombres[i] for i in orden], importancias[orden], color="#2b6cb0")
    ax.set_xlabel("Importancia")
    ax.set_title(titulo)
    return _guardar(fig, destino)


def comparacion_modelos(
    nombres: list[str],
    valores: list[float],
    metrica: str,
    destino: Path,
    referencia: float | None = None,
) -> Path:
    """Barras con el desempeno de cada modelo y la linea base como referencia."""
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(nombres), 3.8))
    ax.bar(nombres, valores, color="#2b6cb0")
    if referencia is not None:
        ax.axhline(referencia, color="#c53030", ls="--", lw=1.4, label=f"Linea base = {referencia:.3f}")
        ax.legend(fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(metrica)
    ax.set_title(f"Comparacion de modelos ({metrica})")
    for i, v in enumerate(valores):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    return _guardar(fig, destino)


def curvas_entrenamiento(historial: dict, titulo: str, destino: Path) -> Path:
    """
    Perdida y F1 macro por epoca para la CNN.

    La separacion entre la curva de entrenamiento y la de validacion es la
    evidencia grafica del sobreajuste que pide la rubrica.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    axes[0].plot(historial["epoca"], historial["loss_train"], label="Entrenamiento")
    axes[0].plot(historial["epoca"], historial["loss_val"], label="Validacion")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Perdida")
    axes[0].set_title("Perdida total")
    axes[0].legend(fontsize=8)

    axes[1].plot(historial["epoca"], historial["f1_quality_train"], label="Entrenamiento")
    axes[1].plot(historial["epoca"], historial["f1_quality_val"], label="Validacion")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("F1 macro (calidad)")
    axes[1].set_title("F1 macro de calidad")
    axes[1].legend(fontsize=8)

    fig.suptitle(titulo)
    return _guardar(fig, destino)


def galeria_errores(
    imagenes: list[np.ndarray],
    titulos: list[str],
    destino: Path,
    columnas: int = 5,
) -> Path:
    """
    Mosaico con los errores mas confiados del modelo (analisis de errores).

    Mirar las imagenes mal clasificadas es lo que permite distinguir un fallo
    del modelo de un fallo de la etiqueta o de la segmentacion.
    """
    if not imagenes:
        raise ValueError("No hay imagenes para la galeria de errores")

    filas = int(np.ceil(len(imagenes) / columnas))
    fig, axes = plt.subplots(filas, columnas, figsize=(2.3 * columnas, 2.6 * filas))
    axes = np.atleast_1d(axes).ravel()

    for ax, img, titulo in zip(axes, imagenes, titulos):
        ax.imshow(img[..., ::-1])  # BGR -> RGB
        ax.set_title(titulo, fontsize=7)
        ax.axis("off")
    for ax in axes[len(imagenes):]:
        ax.axis("off")

    fig.suptitle("Errores mas confiados en el conjunto de prueba")
    return _guardar(fig, destino)
