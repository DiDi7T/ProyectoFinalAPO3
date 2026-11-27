"""
Inferencia sobre una sola imagen: calidad y tamano de la fruta.

Este modulo es la fase de Despliegue de CRISP-DM. La interfaz grafica
(src/deployment/app.py) solo llama a `Predictor.predecir`; toda la logica de
preprocesamiento vive aqui.

Regla principal: NO reimplementar el preprocesamiento. Cada paso se importa
del mismo modulo que se uso al entrenar:

    lectura y orientacion  -> src.data.ingest_recolectado.normalizar_imagen
    mascara de la fruta    -> piezas de src.features.segmentacion (ver segmentar)
                              (rembg + descarte de piel, como en las fotos
                              propias; si rembg no esta, heuristica de Otsu)
    93 caracteristicas     -> src.features.extractors.extract_features
    letterbox 128 x 128    -> src.features.build_features.a_cuadrado
    normalizacion de la red-> src.models.cnn.normalizar_lote

Si la interfaz preprocesara distinto a como se entreno, el modelo perderia
desempeno sin ningun aviso (training-serving skew).

Uso desde consola
-----------------
    python -m src.deployment.predict ruta/a/foto.jpg
    python -m src.deployment.predict ruta/a/foto.jpg --modelo clasico
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.data.ingest_recolectado import normalizar_imagen
from src.features.build_features import GRIS_RELLENO, a_cuadrado
from src.features.extractors import extract_features
from src.features.segmentacion import diametro_relativo
from src.utils.config import CHECKPOINTS, CNN_IMAGE_SIZE, QUALITY_CLASSES, SIZE_CLASSES

RUTA_CLASICO_CALIDAD = CHECKPOINTS / "mejor_classic_quality.joblib"
RUTA_CLASICO_TAMANO = CHECKPOINTS / "mejor_classic_size.joblib"
RUTA_CNN = CHECKPOINTS / "cnn_mejor.pt"

# Por debajo de esta probabilidad la interfaz advierte que la prediccion no es
# confiable y pide revision humana (docs/03_etica_e_impactos.md).
UMBRAL_CONFIANZA = 0.60

MODELOS = ("cnn", "clasico")


# ============================================================
# 1. LECTURA DE LA IMAGEN
# ============================================================

def leer_imagen(origen: str | Path | bytes) -> np.ndarray:
    """
    Devuelve la imagen en BGR uint8 (formato OpenCV), lista para el modelo.

    Acepta una ruta o los bytes del archivo (lo que entrega Streamlit). Pasa
    por `normalizar_imagen`, la misma funcion que proceso las fotografias
    propias del dataset: corrige la orientacion EXIF, lee HEIC, quita el canal
    alfa y limita el lado mayor a 1024 px.
    """
    if isinstance(origen, (str, Path)):
        datos = Path(origen).read_bytes()
    else:
        datos = bytes(origen)

    rgb = np.asarray(normalizar_imagen(datos))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ============================================================
# 2. RESULTADO
# ============================================================

@dataclass
class Prediccion:
    """Salida de una prediccion. Las probabilidades suman 1."""

    calidad: str
    proba_calidad: dict[str, float]
    tamano: str
    proba_tamano: dict[str, float]
    diametro_relativo: float
    mascara: np.ndarray = field(repr=False)
    modelo: str = ""
    metodo_mascara: str = ""
    tiempo_ms: float = 0.0
    # Motivo por el que la imagen no parece una fruta (None si es valida).
    rechazo: str | None = None

    @property
    def confianza_calidad(self) -> float:
        return self.proba_calidad[self.calidad]

    @property
    def confianza_tamano(self) -> float:
        return self.proba_tamano[self.tamano]

    @property
    def baja_confianza(self) -> bool:
        return self.confianza_calidad < UMBRAL_CONFIANZA

    def como_dict(self) -> dict:
        return {
            "calidad": self.calidad,
            "confianza_calidad": round(self.confianza_calidad, 4),
            "proba_calidad": {k: round(v, 4) for k, v in self.proba_calidad.items()},
            "tamano": self.tamano,
            "confianza_tamano": round(self.confianza_tamano, 4),
            "proba_tamano": {k: round(v, 4) for k, v in self.proba_tamano.items()},
            "diametro_relativo": self.diametro_relativo,
            "baja_confianza": self.baja_confianza,
            "modelo": self.modelo,
            "metodo_mascara": self.metodo_mascara,
            "rechazo": self.rechazo,
            "tiempo_ms": round(self.tiempo_ms, 1),
        }


def _vector_a_dict(proba: np.ndarray, clases: tuple[str, ...]) -> dict[str, float]:
    return {c: float(p) for c, p in zip(clases, proba)}


# ============================================================
# 2b. SEGMENTACION Y VALIDACION DE LA ENTRADA
# ============================================================

# Limites de forma de una fruta "redonda" (limon, naranja, manzana, guayaba):
# en las fotos de referencia la solidez es >= 0,95 y la circularidad >= 0,74.
# El banano no es redondo (0,73 / 0,44) y tiene la misma forma que un pulgar,
# asi que para las formas no redondas se pide ademas que un clasificador de
# ImageNet reconozca una fruta en el recorte (ver es_fruta_imagenet).
SOLIDEZ_MINIMA = 0.85
CIRCULARIDAD_MINIMA = 0.55
AREA_MINIMA = 0.015  # fraccion de la imagen
AREA_MAXIMA = 0.85
BORDE_MAXIMO = 0.02  # fraccion del perimetro de la imagen que puede tocar


def descriptores_forma(mascara: np.ndarray) -> dict[str, float]:
    """Area relativa, solidez, circularidad y contacto con el borde de la imagen."""
    alto, ancho = mascara.shape[:2]
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return {"area": 0.0, "solidez": 0.0, "circularidad": 0.0, "borde": 0.0}
    c = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(c)
    casco = cv2.contourArea(cv2.convexHull(c))
    perimetro = cv2.arcLength(c, True)
    m = mascara > 0
    # Se mira una franja un poco hacia adentro y no el borde exacto: la
    # heuristica de Otsu borra un margen del 2 % antes de segmentar, asi que un
    # objeto que se sale del encuadre nunca llega a tocar el pixel del borde.
    k = max(1, int(0.03 * min(alto, ancho)))
    en_borde = m[k, :].sum() + m[-1 - k, :].sum() + m[:, k].sum() + m[:, -1 - k].sum()
    return {
        "area": area / (alto * ancho),
        "solidez": area / max(casco, 1.0),
        "circularidad": 4 * np.pi * area / max(perimetro**2, 1.0),
        "borde": en_borde / (2 * (alto + ancho)),
    }


def _es_compacta(mascara: np.ndarray) -> bool:
    d = descriptores_forma(mascara)
    return (
        d["solidez"] >= 0.90
        and d["circularidad"] >= 0.60
        and d["borde"] <= BORDE_MAXIMO
        and d["area"] >= AREA_MINIMA
    )


def segmentar(imagen: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Mascara de la fruta para inferencia.

    Usa las mismas piezas que `src.features.segmentacion.mascara_fruta`, con
    un cambio: el descarte de piel solo se aplica si el primer plano de rembg
    NO es ya un objeto compacto. El filtro de piel existe para quitar la mano
    que sostiene la fruta, pero el amarillo de una guayaba, un limon maduro o
    una naranja cae dentro del rango de piel en YCrCb, y el filtro se comia
    la mitad de la fruta (fallo observado en produccion). Si rembg devolvio
    un solo objeto redondo que no toca el borde, no hay mano que quitar.
    """
    from src.data.build_dataset import get_fruit_mask_and_size
    from src.features import segmentacion as sg

    def heuristica():
        return get_fruit_mask_and_size(imagen)[0], "heuristica"

    if not sg.disponible():
        return heuristica()
    try:
        primer_plano = sg.mascara_primer_plano(imagen)
    except Exception:  # noqa: BLE001 - fallo de rembg o de la descarga del modelo
        return heuristica()

    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def limpiar(m: np.ndarray) -> np.ndarray:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, nucleo)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, nucleo)
        return sg.rellenar_huecos(sg._componente_central(m))

    mascara, metodo = limpiar(primer_plano), "rembg"
    if not _es_compacta(mascara):
        sin_piel = cv2.bitwise_and(primer_plano, cv2.bitwise_not(sg.mascara_piel(imagen)))
        if sin_piel.sum() > 0.35 * primer_plano.sum():
            mascara = sg.rellenar_huecos(sg.recuperar_borde(limpiar(sin_piel), primer_plano))
            metodo = "rembg_sin_piel"

    if (mascara > 0).mean() < 0.01:
        return heuristica()
    return mascara, metodo


# Clases de frutas y verduras de ImageNet (indices 937-957: brocoli ... granada,
# incluye banana, orange, lemon, Granny Smith, fig, custard apple) mas 988-990
# (acorn, hip, buckeye), que es como suele etiquetar las frutas manchadas.
CLASES_FRUTA_IMAGENET = list(range(937, 958)) + [988, 989, 990]
UMBRAL_FRUTA_IMAGENET = 0.15

_IMAGENET = None


def _modelo_imagenet():
    """MobileNetV3 preentrenada en ImageNet, cargada una sola vez (o None)."""
    global _IMAGENET
    if _IMAGENET is None:
        try:
            from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

            pesos = MobileNet_V3_Large_Weights.IMAGENET1K_V2
            _IMAGENET = (mobilenet_v3_large(weights=pesos).eval(), pesos.transforms())
        except Exception:  # noqa: BLE001 - sin torchvision o sin internet
            _IMAGENET = False
    return _IMAGENET or None


def puntaje_fruta_imagenet(imagen: np.ndarray, mascara: np.ndarray) -> float | None:
    """
    Probabilidad total de las clases de fruta de ImageNet sobre el recorte de
    la region detectada (con un margen del 15 %). None si no hay modelo.

    Se recorta porque sobre la foto completa el fondo domina: una manzana sobre
    una bandeja sale como "tray".
    """
    modelo = _modelo_imagenet()
    if modelo is None:
        return None
    import torch
    from PIL import Image

    red, transformar = modelo
    x, y, w, h = cv2.boundingRect(mascara)
    margen = int(0.15 * max(w, h))
    recorte = imagen[max(0, y - margen) : y + h + margen, max(0, x - margen) : x + w + margen]
    entrada = transformar(Image.fromarray(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)))[None]
    with torch.no_grad():
        proba = red(entrada).softmax(1)[0]
    return float(proba[CLASES_FRUTA_IMAGENET].sum())


def validar_fruta(mascara: np.ndarray, imagen: np.ndarray | None = None) -> str | None:
    """
    Devuelve None si la region parece una fruta, o el motivo del rechazo.

    El modelo siempre devuelve alguna clase, incluso para una foto de una
    persona. Este filtro evita mostrar una prediccion sin sentido:

    1. Tamano y encuadre: el objeto debe verse completo.
    2. Si la forma es redonda se acepta. Una fruta podrida sigue siendo
       redonda, pero ImageNet no la reconoce (la llama "chiton" o "nail").
    3. Si no es redonda (banano, o un pulgar), se acepta solo si ImageNet ve
       una fruta en el recorte.
    """
    d = descriptores_forma(mascara)
    if d["area"] < AREA_MINIMA:
        return "no se encontró ningún objeto o es demasiado pequeño en la foto"
    if d["area"] > AREA_MAXIMA or d["borde"] > BORDE_MAXIMO:
        return "el objeto se sale del encuadre"
    if d["solidez"] >= SOLIDEZ_MINIMA and d["circularidad"] >= CIRCULARIDAD_MINIMA:
        return None
    if imagen is not None:
        puntaje = puntaje_fruta_imagenet(imagen, mascara)
        if puntaje is not None and puntaje >= UMBRAL_FRUTA_IMAGENET:
            return None
    return "lo que se detectó no parece una fruta"


# ============================================================
# 3. PREDICTOR
# ============================================================

class Predictor:
    """
    Carga los modelos una sola vez y predice imagen por imagen.

    Los modelos que no existan en disco simplemente no quedan disponibles;
    `disponibles` dice cuales se pueden usar.
    """

    def __init__(
        self,
        ruta_cnn: Path = RUTA_CNN,
        ruta_calidad: Path = RUTA_CLASICO_CALIDAD,
        ruta_tamano: Path = RUTA_CLASICO_TAMANO,
        usar_rembg: bool = True,
    ) -> None:
        # Las fotos que sube un usuario se parecen a las fotos propias del
        # dataset (fondos reales, manos), y esas se segmentaron con rembg al
        # entrenar. Por eso se usa rembg por defecto.
        self.usar_rembg = usar_rembg
        self.cnn = None
        self.cnn_config: dict = {}
        self.clasico_calidad = None
        self.clasico_tamano = None

        if Path(ruta_cnn).exists():
            self._cargar_cnn(Path(ruta_cnn))
        if Path(ruta_calidad).exists() and Path(ruta_tamano).exists():
            import joblib

            self.clasico_calidad = joblib.load(ruta_calidad)
            self.clasico_tamano = joblib.load(ruta_tamano)

    def calentar(self) -> None:
        """
        Hace una prediccion de prueba para cargar rembg y los modelos en
        memoria. Sin esto, la primera foto del usuario tarda varios segundos.
        """
        _modelo_imagenet()
        if self.disponibles:
            imagen = np.full((256, 256, 3), 200, dtype=np.uint8)
            cv2.circle(imagen, (128, 128), 60, (0, 140, 255), -1)
            for modelo in self.disponibles:
                self.predecir(imagen, modelo)

    def _cargar_cnn(self, ruta: Path) -> None:
        import torch

        from src.models.cnn import FruitQualityCNN

        punto = torch.load(ruta, map_location="cpu", weights_only=False)
        config = punto.get("config", {})
        modelo = FruitQualityCNN(
            n_clases_calidad=config.get("n_clases_calidad", len(QUALITY_CLASSES)),
            n_clases_tamano=config.get("n_clases_tamano", len(SIZE_CLASSES)),
            dropout=config.get("dropout", 0.3),
        )
        modelo.load_state_dict(punto["state_dict"])
        modelo.eval()
        self.cnn = modelo
        self.cnn_config = config

    @property
    def disponibles(self) -> list[str]:
        # El primero es el que se usa por defecto. El clasico va primero porque
        # tiene mejor F1 macro de calidad en prueba (0,885 frente a 0,830 de la
        # CNN, ver experiments/results/consolidado_quality.csv).
        modelos = []
        if self.clasico_calidad is not None:
            modelos.append("clasico")
        if self.cnn is not None:
            modelos.append("cnn")
        return modelos

    # --------------------------------------------------------

    def predecir(self, origen: str | Path | bytes | np.ndarray, modelo: str | None = None) -> Prediccion:
        """
        Predice calidad y tamano de una imagen.

        Parameters
        ----------
        origen : ruta, bytes del archivo o imagen BGR ya cargada.
        modelo : 'cnn' o 'clasico'. Si es None usa el primero disponible.
        """
        if not self.disponibles:
            raise RuntimeError(
                "No hay modelos entrenados en experiments/checkpoints/. "
                "Ejecuta primero el entrenamiento (ver README)."
            )
        modelo = modelo or self.disponibles[0]
        if modelo not in self.disponibles:
            raise ValueError(f"Modelo '{modelo}' no disponible. Opciones: {self.disponibles}")

        inicio = time.perf_counter()
        imagen = origen if isinstance(origen, np.ndarray) else leer_imagen(origen)
        if self.usar_rembg:
            mascara, metodo_mascara = segmentar(imagen)
        else:
            from src.data.build_dataset import get_fruit_mask_and_size

            mascara, _, _ = get_fruit_mask_and_size(imagen)
            metodo_mascara = "heuristica"
        diametro, _ = diametro_relativo(mascara)

        if modelo == "cnn":
            proba_q, proba_s = self._predecir_cnn(imagen, mascara)
        else:
            proba_q, proba_s = self._predecir_clasico(imagen, mascara)

        tiempo = (time.perf_counter() - inicio) * 1000
        return Prediccion(
            calidad=QUALITY_CLASSES[int(np.argmax(proba_q))],
            proba_calidad=_vector_a_dict(proba_q, QUALITY_CLASSES),
            tamano=SIZE_CLASSES[int(np.argmax(proba_s))],
            proba_tamano=_vector_a_dict(proba_s, SIZE_CLASSES),
            diametro_relativo=float(diametro),
            mascara=mascara,
            modelo=modelo,
            metodo_mascara=metodo_mascara,
            tiempo_ms=tiempo,
            rechazo=validar_fruta(mascara, imagen),
        )

    def _predecir_cnn(self, imagen: np.ndarray, mascara: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch
        import torch.nn.functional as F

        from src.models.cnn import normalizar_lote

        lado = max(CNN_IMAGE_SIZE)
        img = a_cuadrado(imagen, lado, cv2.INTER_AREA, GRIS_RELLENO)
        if self.cnn_config.get("aplicar_mascara"):
            m = a_cuadrado(mascara, lado, cv2.INTER_NEAREST, 0)
            img = img * (m > 0).astype(np.uint8)[..., None]

        x = torch.from_numpy(normalizar_lote(img[None, ...]))
        with torch.no_grad():
            logits_q, logits_s = self.cnn(x)
        return F.softmax(logits_q, dim=1)[0].numpy(), F.softmax(logits_s, dim=1)[0].numpy()

    def _predecir_clasico(self, imagen: np.ndarray, mascara: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = extract_features(imagen, mascara)[None, :]
        return (
            _proba_clasico(self.clasico_calidad, x, len(QUALITY_CLASSES)),
            _proba_clasico(self.clasico_tamano, x, len(SIZE_CLASSES)),
        )


def _proba_clasico(estimador, x: np.ndarray, n_clases: int) -> np.ndarray:
    """
    Probabilidades en el orden global de clases.

    El estimador solo conoce las clases que vio al entrenar (`classes_`), que
    pueden ser menos de n_clases. Se reubican en un vector de largo fijo. Si el
    modelo no da probabilidades (p. ej. SVM sin probability=True), se devuelve
    1 para la clase predicha.
    """
    proba = np.zeros(n_clases, dtype=np.float64)
    clases = np.asarray(getattr(estimador, "classes_", np.arange(n_clases))).astype(int)
    if hasattr(estimador, "predict_proba"):
        try:
            proba[clases] = estimador.predict_proba(x)[0]
            return proba
        except (AttributeError, ValueError):
            pass
    proba[int(estimador.predict(x)[0])] = 1.0
    return proba


# ============================================================
# 4. CONSOLA
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Predice calidad y tamano de una fruta.")
    parser.add_argument("imagen", type=Path)
    parser.add_argument("--modelo", choices=MODELOS, default=None)
    args = parser.parse_args()

    resultado = Predictor().predecir(args.imagen, args.modelo)
    for clave, valor in resultado.como_dict().items():
        print(f"{clave:>18}: {valor}")


if __name__ == "__main__":
    main()
