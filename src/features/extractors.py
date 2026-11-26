"""
Extraccion de caracteristicas interpretables para los modelos clasicos.

Motivacion
----------
El extractor inicial del equipo (src.data.preprocess.extract_ml_features)
usa un histograma HSV conjunto de 16x8x8 = 1024 bins mas 4 descriptores de
forma. Con ~2.500 imagenes etiquetadas eso deja una razon
muestras/dimensiones cercana a 2.4, un regimen en el que los modelos de
margenes o de arboles sobreajustan con facilidad (maldicion de la
dimensionalidad).

Este modulo define un vector mucho mas compacto (93 dimensiones) en el que
cada bloque tiene una justificacion fisica respecto al problema de calidad:

  1. Color marginal   (64)  Histogramas 1D de H, S y V dentro de la mascara.
  2. Momentos de color (9)  Media, desviacion y asimetria por canal.
  3. Forma             (6)  Diametro equivalente, razon de aspecto, extension,
                            solidez, circularidad y excentricidad.
  4. Textura          (10)  LBP rotacion-invariante uniforme (P=8, R=1).
  5. Defectos          (4)  Proxies de manchas/golpes: fraccion de pixeles
                            oscuros, fraccion de tonos pardos, contraste
                            local y entropia del canal V.

Las ecuaciones de cada bloque estan documentadas en
docs/02_formulacion_matematica.md.

Referencia de la textura: T. Ojala, M. Pietikainen y T. Maenpaa,
"Multiresolution gray-scale and rotation invariant texture classification
with local binary patterns", IEEE TPAMI, vol. 24, no. 7, pp. 971-987, 2002.
"""

from __future__ import annotations

import cv2
import numpy as np

BINS_H, BINS_S, BINS_V = 32, 16, 16
N_COLOR_HIST = BINS_H + BINS_S + BINS_V
N_COLOR_MOMENTS = 9
N_SHAPE = 6
N_LBP = 10
N_DEFECT = 4
N_FEATURES = N_COLOR_HIST + N_COLOR_MOMENTS + N_SHAPE + N_LBP + N_DEFECT

EPS = 1e-8


def feature_names() -> list[str]:
    """Nombres de las 93 caracteristicas, en el mismo orden que el vector."""
    nombres: list[str] = []
    nombres += [f"hist_H_{i:02d}" for i in range(BINS_H)]
    nombres += [f"hist_S_{i:02d}" for i in range(BINS_S)]
    nombres += [f"hist_V_{i:02d}" for i in range(BINS_V)]
    for canal in ("H", "S", "V"):
        nombres += [f"mean_{canal}", f"std_{canal}", f"skew_{canal}"]
    nombres += [
        "diametro_relativo",
        "razon_aspecto",
        "extension",
        "solidez",
        "circularidad",
        "excentricidad",
    ]
    nombres += [f"lbp_riu2_{i}" for i in range(N_LBP)]
    nombres += ["frac_oscuros", "frac_pardos", "contraste_local", "entropia_V"]
    return nombres


# ============================================================
# 1. COLOR
# ============================================================

def _histogramas_marginales(hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Histogramas 1D normalizados de H, S y V calculados solo sobre la mascara.

    Cada histograma se normaliza por el numero de pixeles de la mascara:

        p_k = n_k / sum_j n_j

    de modo que el vector es invariante al tamano en pixeles de la fruta.
    """
    partes = []
    for canal, bins, rango in (
        (0, BINS_H, [0, 180]),
        (1, BINS_S, [0, 256]),
        (2, BINS_V, [0, 256]),
    ):
        hist = cv2.calcHist([hsv], [canal], mask, [bins], rango).flatten()
        total = hist.sum()
        partes.append(hist / total if total > 0 else hist)
    return np.concatenate(partes).astype(np.float32)


def _momentos_color(hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Media, desviacion estandar y asimetria de cada canal dentro de la mascara:

        mu   = (1/N) sum x_i
        sig  = sqrt( (1/N) sum (x_i - mu)^2 )
        skew = (1/N) sum ((x_i - mu)/sig)^3

    La asimetria del canal V distingue frutas con manchas oscuras locales
    (cola izquierda larga) de frutas uniformemente iluminadas.
    """
    pix = hsv[mask > 0].astype(np.float32)
    if pix.size == 0:
        return np.zeros(N_COLOR_MOMENTS, dtype=np.float32)

    salida: list[float] = []
    for c in range(3):
        x = pix[:, c]
        mu = float(x.mean())
        sig = float(x.std())
        skew = float((((x - mu) / (sig + EPS)) ** 3).mean())
        escala = 180.0 if c == 0 else 255.0
        salida += [mu / escala, sig / escala, skew]
    return np.asarray(salida, dtype=np.float32)


# ============================================================
# 2. FORMA
# ============================================================

def _descriptores_forma(mask: np.ndarray) -> np.ndarray:
    """
    Seis descriptores geometricos adimensionales del contorno principal.

        d_rel         = sqrt(4A/pi) / max(H, W)
        razon_aspecto = w / h
        extension     = A / (w*h)
        solidez       = A / A_convexa
        circularidad  = 4*pi*A / P^2
        excentricidad = sqrt(1 - (b/a)^2)

    La circularidad y la excentricidad separan bananos (alargados) de
    citricos (casi circulares) y penalizan contornos deformados por golpes.
    """
    h, w = mask.shape[:2]
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return np.zeros(N_SHAPE, dtype=np.float32)

    c = max(contornos, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perimetro = float(cv2.arcLength(c, True))
    _, _, bw, bh = cv2.boundingRect(c)

    d_equivalente = float(np.sqrt(4.0 * area / np.pi))
    d_rel = d_equivalente / max(h, w)
    razon_aspecto = bw / bh if bh > 0 else 1.0
    extension = area / (bw * bh) if bw * bh > 0 else 0.0

    casco = cv2.convexHull(c)
    area_casco = float(cv2.contourArea(casco))
    solidez = area / area_casco if area_casco > 0 else 0.0

    circularidad = 4.0 * np.pi * area / (perimetro**2) if perimetro > 0 else 0.0

    excentricidad = 0.0
    if len(c) >= 5:
        (_, _), (eje_menor, eje_mayor), _ = cv2.fitEllipse(c)
        a, b = max(eje_mayor, eje_menor), min(eje_mayor, eje_menor)
        if a > 0:
            excentricidad = float(np.sqrt(max(0.0, 1.0 - (b / a) ** 2)))

    return np.asarray(
        [d_rel, razon_aspecto, extension, solidez, circularidad, excentricidad],
        dtype=np.float32,
    )


# ============================================================
# 3. TEXTURA: LBP rotacion-invariante uniforme (P = 8, R = 1)
# ============================================================

_DESPLAZAMIENTOS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _tabla_riu2() -> np.ndarray:
    """
    Tabla de 256 entradas que mapea cada patron LBP de 8 bits a su etiqueta
    rotacion-invariante uniforme (riu2):

        U(p) = numero de transiciones de 0 a 1 y de 1 a 0 en el patron
               circular. Si U <= 2 la etiqueta es el numero de unos (0..8);
               en cualquier otro caso la etiqueta es 9 (patron no uniforme).
    """
    tabla = np.empty(256, dtype=np.uint8)
    for valor in range(256):
        bits = [(valor >> i) & 1 for i in range(8)]
        transiciones = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        tabla[valor] = sum(bits) if transiciones <= 2 else 9
    return tabla


_TABLA_RIU2 = _tabla_riu2()


def _histograma_lbp(gris: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Histograma normalizado de 10 bins de LBP riu2 dentro de la mascara."""
    centro = gris.astype(np.int16)
    codigo = np.zeros(gris.shape, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(_DESPLAZAMIENTOS):
        vecino = np.roll(np.roll(centro, dy, axis=0), dx, axis=1)
        codigo |= (vecino >= centro).astype(np.uint8) << bit

    etiquetas = _TABLA_RIU2[codigo]

    # Se descarta el borde de 1 pixel, contaminado por el wrap-around de
    # np.roll, y se restringe el histograma al interior de la mascara.
    valida = np.zeros(mask.shape, dtype=bool)
    valida[1:-1, 1:-1] = True
    valida &= mask > 0

    if not valida.any():
        return np.zeros(N_LBP, dtype=np.float32)

    hist = np.bincount(etiquetas[valida].ravel(), minlength=N_LBP).astype(np.float32)
    return hist / hist.sum()


# ============================================================
# 4. PROXIES DE DEFECTO
# ============================================================

def _descriptores_defecto(hsv: np.ndarray, gris: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Cuatro indicadores pensados para golpes y podredumbre.

        frac_oscuros    = |{V < 0.5*mediana(V)}| / N
        frac_pardos     = |{H en [10,25] y S < 150}| / N
        contraste_local = std del gradiente Sobel dentro de la mascara / 255
        entropia_V      = -sum p_k log2(p_k) del histograma de V (32 bins) / 5

    Una fruta sana tiene superficie homogenea: entropia y contraste bajos.
    Una fruta golpeada rompe esa homogeneidad aunque su color medio apenas
    cambie, asi que estos valores aportan informacion que los momentos de
    color no capturan.
    """
    dentro = mask > 0
    n = int(dentro.sum())
    if n == 0:
        return np.zeros(N_DEFECT, dtype=np.float32)

    v = hsv[..., 2][dentro].astype(np.float32)
    h_canal = hsv[..., 0][dentro].astype(np.float32)
    s_canal = hsv[..., 1][dentro].astype(np.float32)

    mediana_v = float(np.median(v))
    frac_oscuros = float((v < 0.5 * mediana_v).sum()) / n
    frac_pardos = float(((h_canal >= 10) & (h_canal <= 25) & (s_canal < 150)).sum()) / n

    gx = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=3)
    magnitud = np.sqrt(gx**2 + gy**2)
    contraste_local = float(magnitud[dentro].std() / 255.0)

    hist, _ = np.histogram(v, bins=32, range=(0, 256))
    p = hist.astype(np.float32) / max(hist.sum(), 1)
    p = p[p > 0]
    entropia = float(-(p * np.log2(p)).sum()) / 5.0

    return np.asarray([frac_oscuros, frac_pardos, contraste_local, entropia], dtype=np.float32)


# ============================================================
# 5. API PUBLICA
# ============================================================

def extract_features(image_bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Devuelve el vector de 93 caracteristicas de una imagen BGR.

    Parameters
    ----------
    image_bgr : imagen en formato OpenCV (BGR, uint8).
    mask : mascara binaria opcional de la fruta. Si es None se calcula con la
        heuristica de saturacion + Otsu de src.data.build_dataset. Cuando
        existe una mascara de rembg (fotografias recolectadas) conviene
        pasarla explicitamente: es mucho mas precisa en fondos con textura.
    """
    if mask is None:
        from src.data.build_dataset import get_fruit_mask_and_size

        mask, _, _ = get_fruit_mask_and_size(image_bgr)

    if mask.shape[:2] != image_bgr.shape[:2]:
        mask = cv2.resize(
            mask,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    mask = (mask > 0).astype(np.uint8) * 255

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gris = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    vector = np.concatenate(
        [
            _histogramas_marginales(hsv, mask),
            _momentos_color(hsv, mask),
            _descriptores_forma(mask),
            _histograma_lbp(gris, mask),
            _descriptores_defecto(hsv, gris, mask),
        ]
    ).astype(np.float32)

    # Red de seguridad: ningun NaN o inf debe llegar a scikit-learn.
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
