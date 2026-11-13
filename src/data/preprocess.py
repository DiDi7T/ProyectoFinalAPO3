
"""Preprocesamiento de imágenes para el proyecto final de APO 3.

Las imágenes se manejan en RGB.
Antes de normalizar: uint8, valores de 0 a 255.
Después de normalizar: float32, valores de 0 a 1.

La segmentación funciona principalmente con fondos uniformes.
Debe validarse visualmente antes de utilizarse con todo el dataset.
"""

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 1. CARGA Y VALIDACIÓN DE IMÁGENES
# ============================================================

def load_image_rgb(image_path):
    """
    Carga una imagen y la convierte de BGR a RGB.

    Retorna:
        Imagen RGB con valores enteros entre 0 y 255.
    """

    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"No existe la imagen: {path}"
        )

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError(
            f"No se pudo leer la imagen: {path}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    return image_rgb


def _validate_rgb(image):
    """
    Verifica que una imagen tenga un formato RGB válido.
    """

    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError(
            "La imagen debe tener forma (alto, ancho, 3)."
        )

    if (
        image.shape[0] == 0
        or image.shape[1] == 0
        or image.dtype != np.uint8
    ):
        raise ValueError(
            "La imagen debe ser uint8 y no estar vacía."
        )


# ============================================================
# 2. REDIMENSIONAMIENTO SIN DEFORMAR
# ============================================================

def resize_with_padding(
    image_rgb,
    target_size=(128, 128),
    padding_color=None
):
    """
    Redimensiona una imagen conservando su proporción.

    Si queda espacio libre, agrega un relleno.

    target_size:
        (ancho, alto)
    """

    _validate_rgb(image_rgb)

    target_w, target_h = map(int, target_size)

    if target_w <= 0 or target_h <= 0:
        raise ValueError(
            "Las dimensiones deben ser positivas."
        )

    h, w = image_rgb.shape[:2]

    # Escala única para evitar deformaciones
    scale = min(
        target_w / w,
        target_h / h
    )

    new_w = max(
        1,
        min(target_w, int(round(w * scale)))
    )

    new_h = max(
        1,
        min(target_h, int(round(h * scale)))
    )

    # INTER_AREA se recomienda al reducir imágenes
    interpolation = (
        cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        image_rgb,
        (new_w, new_h),
        interpolation=interpolation
    )

    # Obtener un color de relleno similar al fondo
    if padding_color is None:

        border = np.concatenate(
            [
                image_rgb[0, :, :],
                image_rgb[-1, :, :],
                image_rgb[:, 0, :],
                image_rgb[:, -1, :]
            ],
            axis=0
        )

        padding_color = tuple(
            int(v)
            for v in np.median(border, axis=0)
        )

    else:

        if (
            len(padding_color) != 3
            or any(
                not 0 <= v <= 255
                for v in padding_color
            )
        ):
            raise ValueError(
                "padding_color debe contener tres valores RGB."
            )

        padding_color = tuple(
            int(v) for v in padding_color
        )

    # Calcular relleno necesario
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top

    result = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=padding_color
    )

    return result


# ============================================================
# 3. NORMALIZACIÓN
# ============================================================

def resize_and_normalize(
    image_path,
    target_size=(128, 128),
    keep_aspect_ratio=True
):
    """
    Carga una fotografía, la redimensiona y normaliza.

    Retorna:
        Imagen RGB float32 con valores entre 0 y 1.

    Conserva el nombre de la función original
    para mantener compatibilidad con el EDA.
    """

    image_rgb = load_image_rgb(image_path)

    if keep_aspect_ratio:

        resized = resize_with_padding(
            image_rgb,
            target_size
        )

    else:

        target_w, target_h = map(
            int,
            target_size
        )

        if target_w <= 0 or target_h <= 0:
            raise ValueError(
                "Las dimensiones deben ser positivas."
            )

        resized = cv2.resize(
            image_rgb,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA
        )

    normalized = (
        resized.astype(np.float32) / 255.0
    )

    return normalized


# ============================================================
# 4. SEGMENTACIÓN EXPERIMENTAL DE FRUTAS
# ============================================================

def segment_fruit_uniform_background(
    image_rgb,
    threshold=25.0,
    min_area_ratio=0.02
):
    """
    Estima una máscara de una fruta sobre fondo uniforme.

    0   = fondo
    255 = fruta

    Utiliza diferencias de color en el espacio LAB,
    operaciones morfológicas y componentes conectados.

    No garantiza segmentación correcta en todas las fotos.
    """

    _validate_rgb(image_rgb)

    if (
        threshold <= 0
        or not (0 < min_area_ratio < 1)
    ):
        raise ValueError(
            "Los parámetros de segmentación son inválidos."
        )

    h, w = image_rgb.shape[:2]

    # Convertir la imagen al espacio de color LAB
    lab = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    # Obtener muestras de los bordes
    border = np.concatenate(
        [
            lab[0, :, :],
            lab[-1, :, :],
            lab[:, 0, :],
            lab[:, -1, :]
        ],
        axis=0
    )

    # Estimar el color del fondo
    background = np.median(
        border,
        axis=0
    )

    # Distancia entre cada píxel y el fondo estimado
    distance = np.linalg.norm(
        lab - background,
        axis=2
    )

    # Crear máscara binaria
    candidate = (
        np.uint8(distance >= threshold) * 255
    )

    # Eliminar ruido y pequeños huecos
    kernel_size = (
        3 if min(h, w) < 180 else 5
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        kernel
    )

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Detectar regiones conectadas
    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            candidate,
            8
        )
    )

    if count < 2:
        raise ValueError(
            "No se detectó una fruta separable del fondo."
        )

    areas = stats[
        1:,
        cv2.CC_STAT_AREA
    ]

    biggest = (
        1 + int(np.argmax(areas))
    )

    # Detectar otras regiones grandes
    significant_other = sum(
        a >= max(
            min_area_ratio * h * w,
            0.2 * areas[biggest - 1]
        )
        for i, a in enumerate(areas, start=1)
        if i != biggest
    )

    if significant_other:
        raise ValueError(
            "Se detectaron varias regiones grandes: "
            "revisar y separar las frutas individualmente."
        )

    mask = (
        np.uint8(labels == biggest) * 255
    )

    # Rellenar huecos interiores
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled = np.zeros_like(mask)

    cv2.drawContours(
        filled,
        contours,
        -1,
        255,
        thickness=cv2.FILLED
    )

    # Comprobar proporción detectada
    area_ratio = (
        np.count_nonzero(filled) / (h * w)
    )

    if not (
        min_area_ratio <= area_ratio <= 0.95
    ):
        raise ValueError(
            "Segmentación poco confiable: "
            f"área detectada {area_ratio:.1%}."
        )

    return filled


# ============================================================
# 5. RECORTE DE LA FRUTA
# ============================================================

def crop_to_foreground(
    image_rgb,
    mask,
    margin_ratio=0.05
):
    """
    Recorta la región de la imagen que contiene
    la fruta y conserva un margen alrededor.
    """

    _validate_rgb(image_rgb)

    if (
        mask.shape != image_rgb.shape[:2]
        or not np.any(mask)
    ):
        raise ValueError(
            "La máscara es inválida."
        )

    if not 0 <= margin_ratio <= 1:
        raise ValueError(
            "margin_ratio debe estar entre 0 y 1."
        )

    ys, xs = np.where(mask > 0)

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1

    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    mx = int(
        round((x1 - x0) * margin_ratio)
    )

    my = int(
        round((y1 - y0) * margin_ratio)
    )

    x0 = max(0, x0 - mx)
    x1 = min(image_rgb.shape[1], x1 + mx)

    y0 = max(0, y0 - my)
    y1 = min(image_rgb.shape[0], y1 + my)

    cropped_image = image_rgb[
        y0:y1,
        x0:x1
    ].copy()

    cropped_mask = mask[
        y0:y1,
        x0:x1
    ].copy()

    return cropped_image, cropped_mask


# ============================================================
# 6. EXTRACCIÓN DE CARACTERÍSTICAS DE COLOR
# ============================================================

def extract_color_histogram(
    image,
    bins=(8, 8, 8),
    mask=None
):
    """
    Obtiene un histograma RGB tridimensional.

    Con 8 intervalos por canal:
        8 * 8 * 8 = 512 características.

    Normaliza para que la suma de frecuencias sea 1.
    """

    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise ValueError(
            "La imagen debe ser RGB y no estar vacía."
        )

    if (
        len(bins) != 3
        or any(int(b) <= 0 for b in bins)
    ):
        raise ValueError(
            "bins debe contener tres enteros positivos."
        )

    bins = tuple(int(b) for b in bins)

    # Convertir al tipo requerido por OpenCV
    if image.dtype == np.uint8:

        img_uint8 = image

    elif np.issubdtype(
        image.dtype,
        np.floating
    ):

        if (
            not np.all(np.isfinite(image))
            or image.min() < 0
            or image.max() > 1
        ):
            raise ValueError(
                "La imagen float debe estar en [0,1]."
            )

        img_uint8 = np.clip(
            np.rint(image * 255),
            0,
            255
        ).astype(np.uint8)

    else:
        raise ValueError(
            "La imagen debe ser uint8 o float normalizada."
        )

    # Validación opcional de la máscara
    if mask is not None:

        if (
            not isinstance(mask, np.ndarray)
            or mask.shape != image.shape[:2]
        ):
            raise ValueError(
                "La máscara debe coincidir con la imagen."
            )

        mask = (
            np.uint8(mask > 0) * 255
        )

        if not np.any(mask):
            raise ValueError(
                "La máscara no contiene píxeles activos."
            )

    # Calcular histograma
    hist = cv2.calcHist(
        [img_uint8],
        [0, 1, 2],
        mask,
        bins,
        [0, 256, 0, 256, 0, 256]
    )

    # Normalización por suma de frecuencias
    total = float(hist.sum())

    if total <= 0:
        raise ValueError(
            "No se encontraron píxeles para el histograma."
        )

    hist = hist / total

    return hist.flatten().astype(np.float32)


# ============================================================
# 7. PIPELINE PARA MACHINE LEARNING
# ============================================================

def extract_combined_features(
    image_path,
    target_size=(128, 128),
    use_segmentation=False
):
    """
    Extrae características de color de una fotografía.

    use_segmentation=False:
        Histograma de la imagen completa procesada.

    use_segmentation=True:
        Histograma únicamente de píxeles de la fruta.
    """

    if use_segmentation:

        image_rgb = load_image_rgb(image_path)

        mask = segment_fruit_uniform_background(
            image_rgb
        )

        return extract_color_histogram(
            image_rgb,
            mask=mask
        )

    image_normalized = resize_and_normalize(
        image_path,
        target_size
    )

    return extract_color_histogram(
        image_normalized
    )


def build_feature_dataset(
    df,
    use_segmentation=False,
    on_error="raise"
):
    """
    Construye matrices X e y desde un DataFrame.

    Columnas necesarias:
        path: ruta de cada imagen
        quality: etiqueta de calidad

    on_error:
        'raise' = detener al encontrar un error
        'skip'  = omitir imágenes inválidas
    """

    for col in ("path", "quality"):

        if col not in df.columns:
            raise ValueError(
                f"Falta la columna: {col}"
            )

    if on_error not in ("raise", "skip"):
        raise ValueError(
            "on_error debe ser 'raise' o 'skip'."
        )

    vectors = []
    labels = []
    discarded = []

    for path, label in zip(
        df["path"],
        df["quality"]
    ):

        try:

            vector = extract_combined_features(
                path,
                use_segmentation=use_segmentation
            )

        except (OSError, ValueError, cv2.error):

            if on_error == "raise":
                raise

            discarded.append(str(path))
            continue

        vectors.append(vector)
        labels.append(label)

    if not vectors:
        raise ValueError(
            "No se encontraron imágenes válidas."
        )

    if discarded:

        import warnings

        warnings.warn(
            f"Se omitieron {len(discarded)} imágenes inválidas. "
            f"Ejemplo: {discarded[0]}",
            RuntimeWarning,
            stacklevel=2
        )

    X = np.stack(vectors)
    y = np.asarray(labels)

    return X, y


# ============================================================
# 8. DATA AUGMENTATION
# ============================================================

def augment_training_image(
    image_rgb,
    rng=None,
    max_rotation=12,
    brightness_range=(0.85, 1.15)
):
    """
    Genera una variación moderada de la imagen.

    Incluye:
        - Rotación pequeña
        - Cambio moderado de brillo

    SOLO debe utilizarse con datos de entrenamiento.
    """

    _validate_rgb(image_rgb)

    if (
        max_rotation < 0
        or brightness_range[0] <= 0
        or brightness_range[0] > brightness_range[1]
    ):
        raise ValueError(
            "Los parámetros de aumento son inválidos."
        )

    if rng is None:
        rng = np.random.default_rng()

    h, w = image_rgb.shape[:2]

    # Rotación aleatoria
    degrees = float(
        rng.uniform(
            -max_rotation,
            max_rotation
        )
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        degrees,
        1
    )

    rotated = cv2.warpAffine(
        image_rgb,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # Variación moderada de iluminación
    factor = float(
        rng.uniform(*brightness_range)
    )

    augmented = np.uint8(
        np.clip(
            rotated.astype(np.float32) * factor,
            0,
            255
        )
    )

    return augmented
