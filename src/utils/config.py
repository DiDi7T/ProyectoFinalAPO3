"""
Configuración central del proyecto (rutas, taxonomías de etiquetas y semillas).

Este módulo es la única fuente de verdad para:
  * las rutas del proyecto (datos crudos, datos procesados, experimentos),
  * el espacio de etiquetas de las dos tareas supervisadas (calidad y tamaño),
  * la semilla global de reproducibilidad.

Cualquier módulo de features, modelado o evaluación debe importar de aquí
en lugar de redefinir rutas o diccionarios de clases. Así se evita que dos
integrantes del equipo entrenen con codificaciones de clases distintas.
"""

from __future__ import annotations

from pathlib import Path

# ============================================================
# 1. RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_PROCESSED = PROJECT_ROOT / "data_processed"

KAGGLE_DIR = DATA_RAW / "kaggle" / "archive"
RECOLECTADO_DIR = DATA_RAW / "recolectado"

LABELS_CSV = DATA_PROCESSED / "labels.csv"
FEATURES_DIR = DATA_PROCESSED / "features"

EXPERIMENTS = PROJECT_ROOT / "experiments"
CHECKPOINTS = EXPERIMENTS / "checkpoints"
LOGS = EXPERIMENTS / "logs"
RESULTS = EXPERIMENTS / "results"

DOCS = PROJECT_ROOT / "docs"


def ensure_dirs() -> None:
    """Crea las carpetas de salida si no existen (idempotente)."""
    for d in (DATA_PROCESSED, FEATURES_DIR, CHECKPOINTS, LOGS, RESULTS):
        d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. ESPACIO DE ETIQUETAS
# ============================================================
# El enunciado exige al menos 3 categorías de calidad. El dataset público de
# Kaggle solo trae dos (Good / Bad), por lo que la clase intermedia 'regular'
# se obtiene de (a) la carpeta Mixed Quality segmentada y anotada manualmente
# y (b) las fotografías recolectadas por el grupo.
#
# El orden de los diccionarios define el orden de las filas/columnas en las
# matrices de confusión y en los reportes de clasificación.

QUALITY_CLASSES: tuple[str, ...] = ("good", "regular", "bad")
SIZE_CLASSES: tuple[str, ...] = ("pequeño", "mediano", "grande")
FRUIT_CLASSES: tuple[str, ...] = ("apple", "banana", "guava", "lime", "orange")

QUALITY_TO_IDX = {c: i for i, c in enumerate(QUALITY_CLASSES)}
IDX_TO_QUALITY = {i: c for c, i in QUALITY_TO_IDX.items()}

SIZE_TO_IDX = {c: i for i, c in enumerate(SIZE_CLASSES)}
IDX_TO_SIZE = {i: c for c, i in SIZE_TO_IDX.items()}

FRUIT_TO_IDX = {c: i for i, c in enumerate(FRUIT_CLASSES)}
IDX_TO_FRUIT = {i: c for c, i in FRUIT_TO_IDX.items()}

# Umbrales del diámetro equivalente normalizado d_rel usados para discretizar
# el tamaño. Documentados en docs/02_formulacion_matematica.md: son válidos
# solo bajo el protocolo de captura a distancia fija descrito allí.
SIZE_THRESHOLDS = (0.35, 0.58)


# ============================================================
# 3. REPRODUCIBILIDAD
# ============================================================

RANDOM_STATE = 42

# Particiones: 70 % entrenamiento, 15 % validación, 15 % prueba.
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Tamaño de entrada de la CNN (alto, ancho).
CNN_IMAGE_SIZE = (128, 128)


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Fija las semillas de random, numpy y torch (si está disponible)."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:  # torch es opcional para los modelos clásicos
        pass
