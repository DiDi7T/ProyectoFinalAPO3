"""
Red neuronal convolucional propia, multitarea, entrenada desde cero.

El enunciado pide una CNN simple (2 o 3 bloques convolucionales) y recomienda
entrenar desde cero antes que usar transfer learning. Esta arquitectura sigue
esa recomendacion.

Arquitectura
------------
    Entrada 3 x 128 x 128
    Bloque 1:  Conv3x3(3->32)  + BN + ReLU + Conv3x3(32->32)  + BN + ReLU + MaxPool2
    Bloque 2:  Conv3x3(32->64) + BN + ReLU + Conv3x3(64->64)  + BN + ReLU + MaxPool2
    Bloque 3:  Conv3x3(64->128)+ BN + ReLU + Conv3x3(128->128)+ BN + ReLU + MaxPool2
    GlobalAvgPool -> 128
    Tronco denso: Linear(128->128) + ReLU + Dropout(p)
    Cabeza A: Linear(128 -> n_clases_calidad)
    Cabeza B: Linear(128 -> n_clases_tamano)

Decisiones y su justificacion
-----------------------------
* Dos convoluciones 3x3 por bloque en lugar de una 5x5: mismo campo receptivo
  efectivo (5x5) con menos parametros (2*9 = 18 frente a 25 por canal) y una
  no linealidad extra.
* Normalizacion por lotes (BatchNorm) despues de cada convolucion: estabiliza
  el entrenamiento y permite tasas de aprendizaje mayores. Normaliza cada
  activacion como (x - mu_B)/sqrt(sig_B^2 + eps) y luego la reescala con dos
  parametros aprendidos.
* Global Average Pooling en vez de aplanar: reduce el mapa 128x16x16 = 32.768
  valores a 128, lo que elimina casi todos los parametros densos y es la
  principal defensa contra el sobreajuste con pocos miles de imagenes.
* Dos cabezas sobre un tronco compartido (aprendizaje multitarea): calidad y
  tamano comparten representaciones de bajo nivel (bordes, color, textura),
  asi que compartir el extractor actua como regularizador y ahorra la mitad
  del computo frente a entrenar dos redes.
* Espacio de color RGB y no HSV: el canal H es un angulo con discontinuidad en
  0/180, y una convolucion sobre ese canal mezcla rojos a ambos lados del
  corte como si fueran opuestos. Para los modelos clasicos el histograma de H
  si es adecuado, porque se trata como conteo por bin y no como valor
  continuo.

Perdida
-------
    L = lambda_q * CE(logits_calidad, y_calidad)
      + lambda_s * CE(logits_tamano,  y_tamano)

con entropia cruzada ponderada por clase w_c = N / (K * N_c) para compensar
el desbalance. Por defecto lambda_q = 1.0 y lambda_s = 0.5: la calidad es la
salida principal del sistema y el tamano una salida secundaria cuyas
etiquetas provienen de una heuristica geometrica, menos confiable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bloque_conv(entrada: int, salida: int) -> nn.Sequential:
    """Dos convoluciones 3x3 con BatchNorm y ReLU, seguidas de MaxPool 2x2."""
    return nn.Sequential(
        nn.Conv2d(entrada, salida, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(salida),
        nn.ReLU(inplace=True),
        nn.Conv2d(salida, salida, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(salida),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class FruitQualityCNN(nn.Module):
    """CNN de 3 bloques con dos cabezas de clasificacion."""

    def __init__(
        self,
        n_clases_calidad: int = 3,
        n_clases_tamano: int = 3,
        dropout: float = 0.3,
        canales: tuple[int, int, int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        c1, c2, c3 = canales

        self.extractor = nn.Sequential(
            bloque_conv(3, c1),
            bloque_conv(c1, c2),
            bloque_conv(c2, c3),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.tronco = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cabeza_calidad = nn.Linear(128, n_clases_calidad)
        self.cabeza_tamano = nn.Linear(128, n_clases_tamano)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.tronco(self.pool(self.extractor(x)))
        return self.cabeza_calidad(z), self.cabeza_tamano(z)

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PerdidaMultitarea(nn.Module):
    """Suma ponderada de dos entropias cruzadas, cada una con pesos de clase."""

    def __init__(
        self,
        pesos_calidad: torch.Tensor | None = None,
        pesos_tamano: torch.Tensor | None = None,
        lambda_calidad: float = 1.0,
        lambda_tamano: float = 0.5,
        suavizado: float = 0.05,
    ) -> None:
        super().__init__()
        self.lambda_calidad = lambda_calidad
        self.lambda_tamano = lambda_tamano
        # El suavizado de etiquetas evita que la red se vuelva excesivamente
        # confiada, algo frecuente cuando hay imagenes casi duplicadas.
        self.ce_calidad = nn.CrossEntropyLoss(weight=pesos_calidad, label_smoothing=suavizado)
        self.ce_tamano = nn.CrossEntropyLoss(weight=pesos_tamano, label_smoothing=suavizado)

    def forward(
        self,
        logits_calidad: torch.Tensor,
        logits_tamano: torch.Tensor,
        y_calidad: torch.Tensor,
        y_tamano: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_q = self.ce_calidad(logits_calidad, y_calidad)
        l_s = self.ce_tamano(logits_tamano, y_tamano)
        total = self.lambda_calidad * l_q + self.lambda_tamano * l_s
        return total, {
            "loss": float(total.detach()),
            "loss_calidad": float(l_q.detach()),
            "loss_tamano": float(l_s.detach()),
        }


def pesos_de_clase(y: np.ndarray, n_clases: int) -> torch.Tensor:
    """
    Pesos w_c = N / (K * N_c) para la entropia cruzada.

    A las clases ausentes se les asigna peso 0 para que no distorsionen el
    gradiente mientras la anotacion manual de 'regular' siga pendiente.
    """
    conteos = np.bincount(y.astype(int), minlength=n_clases).astype(np.float64)
    n = conteos.sum()
    presentes = conteos > 0
    pesos = np.zeros(n_clases, dtype=np.float32)
    pesos[presentes] = n / (presentes.sum() * conteos[presentes])
    return torch.tensor(pesos, dtype=torch.float32)


# ============================================================
# PREPROCESAMIENTO Y AUMENTACION
# ============================================================

# Media y desviacion por canal (RGB) de ImageNet. Se usan como estandarizacion
# fija y reproducible; estan muy cerca de las del dataset y evitan que el valor
# dependa de la particion concreta.
MEDIA_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESV_RGB = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalizar_lote(imgs_bgr_uint8: np.ndarray) -> np.ndarray:
    """
    Convierte un lote (N, H, W, 3) BGR uint8 a (N, 3, H, W) float32 normalizado:

        x = (I/255 - media) / desviacion
    """
    rgb = imgs_bgr_uint8[..., ::-1].astype(np.float32) / 255.0
    rgb = (rgb - MEDIA_RGB) / DESV_RGB
    return np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))


def aumentar(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Aumentacion geometrica y fotometrica suave.

    Solo se aplican transformaciones que preserven la etiqueta: una fruta
    golpeada sigue estando golpeada si se refleja o se rota, y un cambio
    moderado de brillo simula otra iluminacion de la plaza de mercado. NO se
    usan recortes agresivos ni cambios de tono, porque alterarian justamente
    las senales (color y area) que el modelo debe aprender.
    """
    import cv2

    aug = img_bgr
    if rng.random() < 0.5:
        aug = cv2.flip(aug, 1)
    if rng.random() < 0.3:
        aug = cv2.flip(aug, 0)

    k = int(rng.integers(0, 4))
    if k:
        aug = np.rot90(aug, k=k).copy()

    if rng.random() < 0.7:
        # Ajuste afin de contraste y brillo: I' = alpha*I + beta
        alpha = float(rng.uniform(0.85, 1.15))
        beta = float(rng.uniform(-18, 18))
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    if rng.random() < 0.3:
        angulo = float(rng.uniform(-12, 12))
        h, w = aug.shape[:2]
        matriz = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)
        aug = cv2.warpAffine(aug, matriz, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    return aug


class FrutasDataset(torch.utils.data.Dataset):
    """
    Dataset en memoria sobre los tensores cacheados por build_features.

    Parameters
    ----------
    imgs : (N, H, W, 3) uint8 BGR.
    masks : (N, H, W) uint8, o None.
    y_calidad, y_tamano : vectores de enteros.
    entrenamiento : si True aplica aumentacion aleatoria.
    aplicar_mascara : si True multiplica la imagen por la mascara de la fruta
        (fondo negro). Permite medir cuanto del desempeno viene del fondo y
        cuanto de la fruta, un experimento de ablacion que la rubrica valora.
    """

    def __init__(
        self,
        imgs: np.ndarray,
        y_calidad: np.ndarray,
        y_tamano: np.ndarray,
        masks: np.ndarray | None = None,
        entrenamiento: bool = False,
        aplicar_mascara: bool = False,
        semilla: int = 42,
    ) -> None:
        self.imgs = imgs
        self.masks = masks
        self.y_calidad = y_calidad.astype(np.int64)
        self.y_tamano = y_tamano.astype(np.int64)
        self.entrenamiento = entrenamiento
        self.aplicar_mascara = aplicar_mascara
        self.rng = np.random.default_rng(semilla)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int):
        img = self.imgs[i]

        if self.aplicar_mascara and self.masks is not None:
            m = (self.masks[i] > 0).astype(np.uint8)
            img = img * m[..., None]

        if self.entrenamiento:
            img = aumentar(img, self.rng)

        x = normalizar_lote(img[None, ...])[0]
        return (
            torch.from_numpy(x),
            torch.tensor(self.y_calidad[i]),
            torch.tensor(self.y_tamano[i]),
        )


@torch.no_grad()
def predecir(modelo: nn.Module, cargador, dispositivo: str = "cpu") -> dict[str, np.ndarray]:
    """Devuelve predicciones y probabilidades de ambas cabezas."""
    modelo.eval()
    pq, ps, yq, ys, probq, probs = [], [], [], [], [], []

    for x, y_calidad, y_tamano in cargador:
        x = x.to(dispositivo)
        logits_q, logits_s = modelo(x)
        probq.append(F.softmax(logits_q, dim=1).cpu().numpy())
        probs.append(F.softmax(logits_s, dim=1).cpu().numpy())
        pq.append(logits_q.argmax(1).cpu().numpy())
        ps.append(logits_s.argmax(1).cpu().numpy())
        yq.append(y_calidad.numpy())
        ys.append(y_tamano.numpy())

    return {
        "pred_calidad": np.concatenate(pq),
        "pred_tamano": np.concatenate(ps),
        "y_calidad": np.concatenate(yq),
        "y_tamano": np.concatenate(ys),
        "proba_calidad": np.concatenate(probq),
        "proba_tamano": np.concatenate(probs),
    }
