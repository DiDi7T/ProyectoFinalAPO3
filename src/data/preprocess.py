from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.build_dataset import PROJECT_ROOT, get_fruit_mask_and_size

QUALITY_TO_IDX = {"good": 0, "regular": 1, "bad": 2}
IDX_TO_QUALITY = {v: k for k, v in QUALITY_TO_IDX.items()}

FRUIT_TO_IDX = {"apple": 0, "banana": 1, "guava": 2, "lime": 3, "orange": 4}
SIZE_TO_IDX = {"pequeño": 0, "mediano": 1, "grande": 2}


def load_labels_df(csv_path: Path | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = PROJECT_ROOT / "data_processed" / "labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No existe {csv_path}. Ejecuta primero: python -m src.data.build_dataset"
        )
    return pd.read_csv(csv_path)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide estratificadamente por combinación (fruta + calidad) en train, val y test
    antes de aplicar aumento de datos.
    """
    strat_col = df["fruit"] + "_" + df["quality"]

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=strat_col,
        random_state=random_state,
    )

    val_relative = val_size / (1.0 - test_size)
    strat_train_val = train_val_df["fruit"] + "_" + train_val_df["quality"]

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        stratify=strat_train_val,
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def extract_ml_features(image_bgr: np.ndarray, bins: tuple[int, int, int] = (16, 8, 8)) -> np.ndarray:
    """
    Extrae un vector 1D para modelos de ML tradicional:
    - Histograma 3D en espacio HSV calculado SOLO sobre la máscara de la fruta.
    - 4 características geométricas normalizadas (diámetro relativo, aspect ratio, solidez, extensión).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask, rel_diameter, _ = get_fruit_mask_and_size(image_bgr)

    # Histograma HSV enmascarado (ignora píxeles de fondo)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_flat = hist.flatten()

    # Descriptores de forma sobre el contorno principal
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        _, _, bw, bh = cv2.boundingRect(c)
        aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0
        extent = float(area) / float(bw * bh) if (bw * bh) > 0 else 0.0
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / float(hull_area) if hull_area > 0 else 0.0
    else:
        aspect_ratio, extent, solidity = 1.0, 0.0, 0.0

    shape_features = np.array([rel_diameter, aspect_ratio, extent, solidity], dtype=np.float32)
    return np.concatenate([hist_flat.astype(np.float32), shape_features])


def preprocess_image_for_cnn(image_bgr: np.ndarray, target_size: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Redimensiona, convierte a HSV y normaliza píxeles a [0.0, 1.0] para la CNN."""
    resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] /= 179.0  # Canal H en OpenCV va de 0 a 179
    hsv[..., 1] /= 255.0  # Canal S de 0 a 255
    hsv[..., 2] /= 255.0  # Canal V de 0 a 255
    return hsv


def augment_image_bgr(image_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Aplica aumentación suave (flips, rotación de 90°/180°/270° y ligera variación de brillo)."""
    aug = image_bgr.copy()
    if rng.random() > 0.5:
        aug = cv2.flip(aug, 1)
    if rng.random() > 0.5:
        aug = cv2.flip(aug, 0)

    k_rot = int(rng.integers(0, 4))
    if k_rot > 0:
        aug = np.rot90(aug, k=k_rot)

    # Pequeña perturbación en brillo (canal V)
    hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = rng.uniform(0.9, 1.1)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def prepare_ml_split(
    df: pd.DataFrame,
    augment_minorities: bool = False,
    target_per_quality: int = 300,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye matrices (X_features, y_quality, y_size) a partir de un DataFrame split.
    Si augment_minorities=True (usar SOLO en train), balancea clases con menos muestras (ej. 'regular').
    """
    rng = np.random.default_rng(random_state)
    X_list, y_qual_list, y_size_list = [], [], []

    for _, row in df.iterrows():
        img_path = PROJECT_ROOT / row["filepath"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        X_list.append(extract_ml_features(img))
        y_qual_list.append(QUALITY_TO_IDX[row["quality"]])
        y_size_list.append(SIZE_TO_IDX[row["size_label"]])

    if augment_minorities:
        for qual_name, qual_idx in QUALITY_TO_IDX.items():
            subset = df[df["quality"] == qual_name]
            current_count = len(subset)
            if 0 < current_count < target_per_quality:
                needed = target_per_quality - current_count
                sampled = subset.sample(n=needed, replace=True, random_state=random_state)
                for _, row in sampled.iterrows():
                    img_path = PROJECT_ROOT / row["filepath"]
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    aug_img = augment_image_bgr(img, rng)
                    X_list.append(extract_ml_features(aug_img))
                    y_qual_list.append(qual_idx)
                    y_size_list.append(SIZE_TO_IDX[row["size_label"]])

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_qual_list, dtype=np.int64),
        np.array(y_size_list, dtype=np.int64),
    )


def prepare_cnn_split(
    df: pd.DataFrame,
    target_size: tuple[int, int] = (128, 128),
    augment_minorities: bool = False,
    target_per_quality: int = 300,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construye tensores (X_images, y_quality, y_size) en formato HSV normalizado para la CNN."""
    rng = np.random.default_rng(random_state)
    X_list, y_qual_list, y_size_list = [], [], []

    for _, row in df.iterrows():
        img_path = PROJECT_ROOT / row["filepath"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        X_list.append(preprocess_image_for_cnn(img, target_size=target_size))
        y_qual_list.append(QUALITY_TO_IDX[row["quality"]])
        y_size_list.append(SIZE_TO_IDX[row["size_label"]])

    if augment_minorities:
        for qual_name, qual_idx in QUALITY_TO_IDX.items():
            subset = df[df["quality"] == qual_name]
            current_count = len(subset)
            if 0 < current_count < target_per_quality:
                needed = target_per_quality - current_count
                sampled = subset.sample(n=needed, replace=True, random_state=random_state)
                for _, row in sampled.iterrows():
                    img_path = PROJECT_ROOT / row["filepath"]
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    aug_img = augment_image_bgr(img, rng)
                    X_list.append(preprocess_image_for_cnn(aug_img, target_size=target_size))
                    y_qual_list.append(qual_idx)
                    y_size_list.append(SIZE_TO_IDX[row["size_label"]])

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_qual_list, dtype=np.int64),
        np.array(y_size_list, dtype=np.int64),
    )


# ============================================================
# INTEGRACIÓN DE IMÁGENES SEGMENTADAS CON REMBG
# ============================================================

def load_rembg_recolectado_df() -> pd.DataFrame:
    """
    Relaciona las etiquetas originales con las imágenes
    procesadas mediante rembg.

    Conserva las etiquetas de calidad y tamaño existentes.
    Solo incluye máscaras generadas correctamente.
    """

    # Cargar las etiquetas del proyecto.
    df = load_labels_df()

    # Trabajar únicamente con fotografías recolectadas.
    df = df[df["source"] == "recolectado"].copy()

    # Archivo generado por segmentation_rembg.py.
    ruta_manifest = (
        PROJECT_ROOT
        / "data_processed"
        / "rembg_recolectado_manifest.csv"
    )

    if not ruta_manifest.is_file():
        raise FileNotFoundError(
            "Todavía no existe el manifiesto de rembg. "
            "Debe generarse al ejecutar la segmentación."
        )

    manifest = pd.read_csv(ruta_manifest)

    # Seleccionar segmentaciones técnicamente válidas.
    manifest = manifest[
        manifest["status"] == "generada"
    ].copy()

    columnas = [
        "filepath",
        "mask_filepath",
        "isolated_filepath",
        "cnn_filepath"
    ]

    # Relacionar cada fotografía con sus salidas.
    df = df.merge(
        manifest[columnas],
        on="filepath",
        how="inner",
        validate="one_to_one"
    )

    # Verificar que las rutas procesadas existan.
    for columna in columnas[1:]:
        df = df[
            df[columna].apply(
                lambda ruta: (
                    PROJECT_ROOT / ruta
                ).is_file()
            )
        ].copy()

    return df.reset_index(drop=True)


# ============================================================
# PREPARACIÓN DE DATOS UTILIZANDO SEGMENTACIÓN REMBG
# ============================================================

def _leer_par_rembg(row):
    """
    Carga la fruta aislada y su máscara binaria
    sin ejecutar nuevamente el modelo rembg.
    """

    ruta_imagen = (
        PROJECT_ROOT / str(row["isolated_filepath"])
    )

    ruta_mascara = (
        PROJECT_ROOT / str(row["mask_filepath"])
    )

    imagen = cv2.imread(
        str(ruta_imagen),
        cv2.IMREAD_COLOR
    )

    mascara = cv2.imread(
        str(ruta_mascara),
        cv2.IMREAD_GRAYSCALE
    )

    if imagen is None or mascara is None:
        raise FileNotFoundError(
            f"Faltan resultados de rembg: "
            f"{ruta_imagen} o {ruta_mascara}"
        )

    if imagen.shape[:2] != mascara.shape:
        raise ValueError(
            "La máscara y la imagen tienen "
            "dimensiones diferentes."
        )

    if not np.any(mascara > 0):
        raise ValueError(
            f"La máscara está vacía: {ruta_mascara}"
        )

    return imagen, mascara


# ============================================================
# 1. CARACTERÍSTICAS PARA MACHINE LEARNING
# ============================================================

def extract_ml_features_rembg(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bins: tuple[int, int, int] = (16, 8, 8)
) -> np.ndarray:
    """
    Calcula características a partir de una fruta segmentada.

    - Histograma HSV: 16 x 8 x 8 = 1024 valores.
    - Geometría: diámetro, relación de aspecto,
      extensión y solidez.

    Total: 1028 características.
    """

    if (
        image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or mask.shape != image_bgr.shape[:2]
    ):
        raise ValueError(
            "La imagen BGR y la máscara no coinciden."
        )

    mascara = np.where(
        mask > 0, 255, 0
    ).astype(np.uint8)

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        raise ValueError(
            "La máscara no contiene objetos."
        )

    contorno = max(
        contornos,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contorno)

    if area <= 0:
        raise ValueError(
            "El contorno no tiene un área válida."
        )

    # Convertir la fruta a HSV.
    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV
    )

    # Histograma únicamente sobre la máscara.
    histograma = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        mascara,
        bins,
        [0, 180, 0, 256, 0, 256]
    )

    # Mantener la normalización L2 de Diana.
    cv2.normalize(
        histograma,
        histograma
    )

    # Características geométricas.
    _, _, ancho, alto = cv2.boundingRect(
        contorno
    )

    relacion_aspecto = (
        ancho / alto if alto else 1.0
    )

    extension = (
        area / (ancho * alto)
        if ancho * alto else 0.0
    )

    area_convexa = cv2.contourArea(
        cv2.convexHull(contorno)
    )

    solidez = (
        area / area_convexa
        if area_convexa else 0.0
    )

    diametro_relativo = (
        np.sqrt(4 * area / np.pi)
        / max(mascara.shape)
    )

    geometria = np.array(
        [
            diametro_relativo,
            relacion_aspecto,
            extension,
            solidez
        ],
        dtype=np.float32
    )

    return np.concatenate([
        histograma.flatten().astype(np.float32),
        geometria
    ])


# ============================================================
# 2. AUMENTACIÓN DE IMAGEN Y MÁSCARA
# ============================================================

def _aumentar_par_rembg(
    imagen,
    mascara,
    rng
):
    """
    Aplica las mismas transformaciones geométricas
    a la fruta y a su máscara binaria.
    """

    imagen = imagen.copy()
    mascara = mascara.copy()

    if rng.random() > 0.5:
        imagen = cv2.flip(imagen, 1)
        mascara = cv2.flip(mascara, 1)

    if rng.random() > 0.5:
        imagen = cv2.flip(imagen, 0)
        mascara = cv2.flip(mascara, 0)

    rotaciones = int(rng.integers(0, 4))

    if rotaciones:
        imagen = np.ascontiguousarray(
            np.rot90(imagen, rotaciones)
        )

        mascara = np.ascontiguousarray(
            np.rot90(mascara, rotaciones)
        )

    # Variación moderada de brillo.
    hsv = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    factor = rng.uniform(0.9, 1.1)

    hsv[..., 2] = np.clip(
        hsv[..., 2] * factor,
        0,
        255
    )

    imagen = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR
    )

    return imagen, mascara


# ============================================================
# 3. PREPARAR MATRICES PARA ML TRADICIONAL
# ============================================================

def prepare_ml_split_rembg(
    df: pd.DataFrame,
    augment_minorities: bool = False,
    target_per_quality: int = 300,
    random_state: int = 42
):
    """
    Prepara X, y_quality e y_size para ML tradicional.

    Utiliza las máscaras almacenadas por rembg.
    La aumentación debe utilizarse solo en train.
    """

    rng = np.random.default_rng(random_state)

    X = []
    y_quality = []
    y_size = []

    def agregar_fotografia(row, aumentar=False):

        imagen, mascara = _leer_par_rembg(row)

        if aumentar:
            imagen, mascara = _aumentar_par_rembg(
                imagen,
                mascara,
                rng
            )

        caracteristicas = extract_ml_features_rembg(
            imagen,
            mascara
        )

        X.append(caracteristicas)

        y_quality.append(
            QUALITY_TO_IDX[row["quality"]]
        )

        y_size.append(
            SIZE_TO_IDX[row["size_label"]]
        )

    # Imágenes originales procesadas.
    for _, fila in df.iterrows():
        agregar_fotografia(fila)

    # Aumento de clases minoritarias.
    if augment_minorities:

        for calidad in QUALITY_TO_IDX:

            subset = df[
                df["quality"] == calidad
            ]

            faltantes = max(
                0,
                target_per_quality - len(subset)
            )

            if faltantes > 0 and not subset.empty:

                seleccion = subset.sample(
                    n=faltantes,
                    replace=True,
                    random_state=random_state
                )

                for _, fila in seleccion.iterrows():
                    agregar_fotografia(
                        fila,
                        aumentar=True
                    )

    if not X:
        raise ValueError(
            "No hay fotografías para preparar."
        )

    return (
        np.stack(X).astype(np.float32),
        np.array(y_quality, dtype=np.int64),
        np.array(y_size, dtype=np.int64)
    )


# ============================================================
# 4. PREPARAR MATRICES PARA CNN
# ============================================================

def prepare_cnn_split_rembg(
    df: pd.DataFrame,
    target_size: tuple[int, int] = (128, 128),
    augment_minorities: bool = False,
    target_per_quality: int = 300,
    random_state: int = 42
):
    """
    Prepara imágenes HSV normalizadas para CNN.

    Utiliza los recortes de rembg de 224x224
    con fondo negro.

    Las imágenes finales tienen dimensiones
    128x128x3, igual que en el código de Diana.
    """

    rng = np.random.default_rng(random_state)

    X = []
    y_quality = []
    y_size = []

    def agregar_fotografia(row, aumentar=False):

        ruta_imagen = (
            PROJECT_ROOT / str(row["cnn_filepath"])
        )

        imagen = cv2.imread(
            str(ruta_imagen),
            cv2.IMREAD_COLOR
        )

        if imagen is None:
            raise FileNotFoundError(
                f"No existe la imagen: {ruta_imagen}"
            )

        if aumentar:
            imagen = augment_image_bgr(
                imagen,
                rng
            )

        imagen_preparada = preprocess_image_for_cnn(
            imagen,
            target_size=target_size
        )

        X.append(imagen_preparada)

        y_quality.append(
            QUALITY_TO_IDX[row["quality"]]
        )

        y_size.append(
            SIZE_TO_IDX[row["size_label"]]
        )

    # Imágenes originales segmentadas.
    for _, fila in df.iterrows():
        agregar_fotografia(fila)

    # Aumento exclusivamente para entrenamiento.
    if augment_minorities:

        for calidad in QUALITY_TO_IDX:

            subset = df[
                df["quality"] == calidad
            ]

            faltantes = max(
                0,
                target_per_quality - len(subset)
            )

            if faltantes > 0 and not subset.empty:

                seleccion = subset.sample(
                    n=faltantes,
                    replace=True,
                    random_state=random_state
                )

                for _, fila in seleccion.iterrows():
                    agregar_fotografia(
                        fila,
                        aumentar=True
                    )

    if not X:
        raise ValueError(
            "No hay fotografías para preparar."
        )

    return (
        np.stack(X).astype(np.float32),
        np.array(y_quality, dtype=np.int64),
        np.array(y_size, dtype=np.int64)
    )

