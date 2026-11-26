"""
Segmentacion de la fruta combinando una red neuronal con reglas de color.

Por que no basta la heuristica de saturacion
--------------------------------------------
`src.data.build_dataset.get_fruit_mask_and_size` asume que la fruta es lo mas
saturado de la escena sobre un fondo neutro. Funciona bien en las imagenes de
estudio de Kaggle y en las fotografias propias tomadas sobre un meson, pero
falla por completo cuando el fondo es mas cromatico que la fruta.

El caso que lo rompio: 33 fotografias de limones deteriorados sostenidos sobre
un cajon lleno de limones sanos. La fruta de interes esta parda y grisacea
(poco saturada) y el fondo es verde intenso (muy saturado), de modo que el
umbral de Otsu selecciona exactamente lo contrario de lo que se busca. En 7 de
cada 8 imagenes revisadas la mascara cubria el fondo.

Estrategia
----------
1. **rembg** (red U^2-Net de deteccion de objeto saliente) separa el primer
   plano del fondo. No usa color sino forma y contexto, asi que un fondo del
   mismo color no la confunde.
2. El primer plano incluye la mano que sostiene la fruta, asi que se descartan
   los pixeles de piel con una regla en el espacio YCrCb, que es el estandar
   para deteccion de piel por ser poco sensible a la luminancia:

       133 <= Cr <= 173   y   77 <= Cb <= 127

3. De lo que queda se toma la componente conexa grande mas cercana al centro,
   con el mismo criterio que usa la heuristica original.
4. Si el resultado es implausible (area menor al 1 % de la imagen), se vuelve
   a la heuristica de saturacion. Nunca se devuelve una mascara vacia sin
   avisar.

El modelo de rembg se descarga la primera vez que se usa (~170 MB) y queda
cacheado en el directorio del usuario.
"""

from __future__ import annotations

import cv2
import numpy as np

UMBRAL_ALPHA = 127
AREA_MINIMA_RELATIVA = 0.01

_SESION_REMBG = None


def _sesion():
    """Crea la sesion de rembg una sola vez (cargar el modelo es costoso)."""
    global _SESION_REMBG
    if _SESION_REMBG is None:
        from rembg import new_session

        _SESION_REMBG = new_session("u2net")
    return _SESION_REMBG


def disponible() -> bool:
    """True si rembg esta instalado."""
    try:
        import rembg  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


# ============================================================
# 1. PRIMER PLANO CON REMBG
# ============================================================

def mascara_primer_plano(image_bgr: np.ndarray) -> np.ndarray:
    """
    Mascara binaria del objeto en primer plano (normalmente mano + fruta).

    Se le pasa la imagen en RGB porque rembg trabaja sobre PIL/numpy en ese
    orden de canales.
    """
    from rembg import remove

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    salida = remove(rgb, session=_sesion(), only_mask=True)
    alpha = np.asarray(salida)
    if alpha.ndim == 3:
        alpha = alpha[..., -1]
    return (alpha > UMBRAL_ALPHA).astype(np.uint8) * 255


# ============================================================
# 2. DESCARTE DE PIEL
# ============================================================

def mascara_piel(image_bgr: np.ndarray) -> np.ndarray:
    """
    Pixeles con color de piel, segun la regla clasica en YCrCb.

    Se usa YCrCb y no HSV porque separa la luminancia (Y) de la crominancia
    (Cr, Cb), de modo que la regla aguanta cambios de iluminacion y distintos
    tonos de piel mucho mejor que un rango de matiz.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))


# ============================================================
# 3. COMPONENTE PRINCIPAL
# ============================================================

def rellenar_huecos(mask: np.ndarray) -> np.ndarray:
    """
    Rellena los huecos interiores de la mascara.

    Es un paso imprescindible despues del descarte de piel. El rango de
    crominancia de la piel se solapa con el de los pardeamientos y las zonas
    podridas, de modo que el filtro abre agujeros justo sobre los defectos:
    precisamente la informacion que el clasificador necesita. Como la fruta es
    un objeto solido y convexo, cualquier hueco interior es un falso negativo
    y se puede recuperar rellenando el contorno externo.

    La mano no se recupera porque, al quitarle los pixeles de piel, deja de
    estar conectada a la fruta y se descarta como componente aparte.
    """
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return mask
    relleno = np.zeros_like(mask)
    cv2.drawContours(relleno, contornos, -1, 255, thickness=cv2.FILLED)
    return relleno


def recuperar_borde(mask: np.ndarray, primer_plano: np.ndarray) -> np.ndarray:
    """
    Recupera las zonas dañadas que el filtro de piel recorto en el borde.

    `rellenar_huecos` solo devuelve lo que quedo rodeado de fruta. Cuando el
    pardeamiento llega hasta el borde (el caso frecuente: la zona podrida esta
    justo donde la mano sostiene la fruta), el recorte deja una mordida que el
    relleno no repara.

    Como la fruta es aproximadamente convexa, se toma la envolvente convexa de
    la componente y se intersecta con el primer plano que devolvio rembg. Esa
    interseccion devuelve la mordida sin incorporar fondo, porque el fondo
    nunca formo parte del primer plano. Solo se acepta si el crecimiento es
    moderado (menos del 60 %), para no tragarse los dedos cuando la mano queda
    por delante de la fruta.
    """
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return mask

    casco = cv2.convexHull(max(contornos, key=cv2.contourArea))
    envolvente = np.zeros_like(mask)
    cv2.drawContours(envolvente, [casco], -1, 255, thickness=cv2.FILLED)

    recuperada = cv2.bitwise_and(envolvente, primer_plano)
    area_antes = max(int((mask > 0).sum()), 1)
    if (recuperada > 0).sum() > 1.6 * area_antes:
        return mask
    return cv2.bitwise_or(mask, recuperada)


def _componente_central(mask: np.ndarray) -> np.ndarray:
    """
    Conserva la componente conexa que maximiza area ponderada por cercania al
    centro, el mismo criterio de `get_fruit_mask_and_size`.
    """
    h, w = mask.shape[:2]
    n, etiquetas, estad, centroides = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask

    centro = np.array([w / 2.0, h / 2.0])
    diagonal = np.hypot(w, h)
    mejor, mejor_puntaje = None, -1.0

    for i in range(1, n):
        area = float(estad[i, cv2.CC_STAT_AREA])
        if area < h * w * AREA_MINIMA_RELATIVA:
            continue
        distancia = float(np.linalg.norm(centroides[i] - centro)) / diagonal
        puntaje = area * (1.0 - 0.7 * distancia)
        if puntaje > mejor_puntaje:
            mejor, mejor_puntaje = i, puntaje

    if mejor is None:
        return mask
    return ((etiquetas == mejor).astype(np.uint8)) * 255


# ============================================================
# 4. API PUBLICA
# ============================================================

def mascara_fruta(image_bgr: np.ndarray, quitar_piel: bool = True) -> tuple[np.ndarray, str]:
    """
    Devuelve (mascara, metodo) con la mejor mascara disponible de la fruta.

    `metodo` vale 'rembg', 'rembg_sin_piel' o 'heuristica', y se registra en el
    manifiesto para poder saber despues como se obtuvo cada mascara.
    """
    from src.data.build_dataset import get_fruit_mask_and_size

    if not disponible():
        mask, _, _ = get_fruit_mask_and_size(image_bgr)
        return mask, "heuristica"

    h, w = image_bgr.shape[:2]
    area_total = h * w

    try:
        mask = mascara_primer_plano(image_bgr)
    except Exception:  # fallo del modelo o de la descarga
        mask, _, _ = get_fruit_mask_and_size(image_bgr)
        return mask, "heuristica"

    primer_plano = mask.copy()
    metodo = "rembg"
    if quitar_piel:
        piel = mascara_piel(image_bgr)
        sin_piel = cv2.bitwise_and(mask, cv2.bitwise_not(piel))
        # Solo se acepta el descarte de piel si queda fruta suficiente: en una
        # naranja o un mango maduro el color puede caer dentro del rango de
        # piel y el filtro se comeria la fruta entera.
        if sin_piel.sum() > 0.35 * mask.sum():
            mask, metodo = sin_piel, "rembg_sin_piel"

    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, nucleo)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, nucleo)
    mask = _componente_central(mask)
    mask = rellenar_huecos(mask)
    if metodo == "rembg_sin_piel":
        mask = rellenar_huecos(recuperar_borde(mask, primer_plano))

    if (mask > 0).sum() < AREA_MINIMA_RELATIVA * area_total:
        mask, _, _ = get_fruit_mask_and_size(image_bgr)
        return mask, "heuristica"

    return mask, metodo


def diametro_relativo(mask: np.ndarray) -> tuple[float, str]:
    """
    Diametro equivalente normalizado y su etiqueta discreta, con los mismos
    umbrales que el resto del proyecto (ver src/utils/config.py).
    """
    from src.utils.config import SIZE_THRESHOLDS

    h, w = mask.shape[:2]
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return 0.5, "mediano"

    area = cv2.contourArea(max(contornos, key=cv2.contourArea))
    d = float(np.sqrt(4 * area / np.pi) / max(h, w))
    d = float(np.clip(d, 0.0, 1.0))

    bajo, alto = SIZE_THRESHOLDS
    if d < bajo:
        etiqueta = "pequeño"
    elif d <= alto:
        etiqueta = "mediano"
    else:
        etiqueta = "grande"
    return round(d, 4), etiqueta
