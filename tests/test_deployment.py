"""
Pruebas del modulo de despliegue con modelos minimos entrenados al vuelo.

No dependen de los checkpoints reales ni del dataset: crean una imagen
sintetica (circulo naranja sobre fondo gris) y modelos pequenos en tmp_path.
"""

from __future__ import annotations

import cv2
import joblib
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.deployment.predict import Predictor, leer_imagen  # noqa: E402
from src.features.extractors import N_FEATURES  # noqa: E402
from src.models.cnn import FruitQualityCNN  # noqa: E402
from src.utils.config import QUALITY_CLASSES, SIZE_CLASSES  # noqa: E402


def imagen_sintetica(alto: int = 480, ancho: int = 360) -> np.ndarray:
    img = np.full((alto, ancho, 3), 200, dtype=np.uint8)
    cv2.circle(img, (ancho // 2, alto // 2), min(alto, ancho) // 4, (0, 140, 255), -1)
    return img


@pytest.fixture(scope="module")
def rutas(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("checkpoints")
    rng = np.random.default_rng(0)

    torch.manual_seed(0)
    cnn = FruitQualityCNN()
    torch.save(
        {"state_dict": cnn.state_dict(), "config": {"dropout": 0.3, "aplicar_mascara": False,
                                                    "n_clases_calidad": 3, "n_clases_tamano": 3}},
        carpeta / "cnn.pt",
    )

    X = rng.normal(size=(60, N_FEATURES))
    # El modelo de calidad solo ve dos clases (como pasa con Kaggle sin 'regular').
    y_q = rng.choice([0, 2], size=60)
    y_s = rng.integers(0, 3, size=60)
    for nombre, y in (("q.joblib", y_q), ("s.joblib", y_s)):
        modelo = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
        joblib.dump(modelo.fit(X, y), carpeta / nombre)

    return carpeta / "cnn.pt", carpeta / "q.joblib", carpeta / "s.joblib"


def test_sin_modelos(tmp_path):
    predictor = Predictor(tmp_path / "x.pt", tmp_path / "x.joblib", tmp_path / "y.joblib")
    assert predictor.disponibles == []
    with pytest.raises(RuntimeError):
        predictor.predecir(imagen_sintetica())


@pytest.mark.parametrize("modelo", ["cnn", "clasico"])
def test_prediccion_valida(rutas, modelo):
    predictor = Predictor(*rutas, usar_rembg=False)
    assert predictor.disponibles == ["clasico", "cnn"]

    r = predictor.predecir(imagen_sintetica(), modelo)
    assert r.calidad in QUALITY_CLASSES
    assert r.tamano in SIZE_CLASSES
    assert set(r.proba_calidad) == set(QUALITY_CLASSES)
    assert sum(r.proba_calidad.values()) == pytest.approx(1.0)
    assert sum(r.proba_tamano.values()) == pytest.approx(1.0)
    assert 0.0 < r.diametro_relativo <= 1.0
    assert r.mascara.shape == (480, 360)
    assert r.modelo == modelo


def test_clase_no_vista_tiene_probabilidad_cero(rutas):
    r = Predictor(*rutas, usar_rembg=False).predecir(imagen_sintetica(), "clasico")
    assert r.proba_calidad["regular"] == 0.0


def test_leer_imagen_desde_bytes():
    ok, codificada = cv2.imencode(".png", imagen_sintetica(2000, 1500))
    assert ok
    img = leer_imagen(codificada.tobytes())
    # normalizar_imagen limita el lado mayor a 1024 px, igual que en la ingesta.
    assert img.shape == (1024, 768, 3)
    # Se devuelve en BGR: el circulo naranja conserva su canal rojo alto.
    assert img[512, 384, 2] > 200


def test_validar_fruta_acepta_circulo_y_rechaza_formas_raras():
    from src.deployment.predict import validar_fruta

    circulo = np.zeros((400, 300), np.uint8)
    cv2.circle(circulo, (150, 200), 80, 255, -1)
    assert validar_fruta(circulo) is None

    vacia = np.zeros((400, 300), np.uint8)
    assert validar_fruta(vacia) is not None

    # Forma de "L", como un brazo o un pulgar.
    ele = np.zeros((400, 300), np.uint8)
    ele[100:300, 100:130] = 255
    ele[270:300, 100:250] = 255
    assert "no parece una fruta" in validar_fruta(ele)

    # Objeto que se sale del encuadre, como una camiseta.
    borde = np.zeros((400, 300), np.uint8)
    borde[150:400, 100:300] = 255
    assert "encuadre" in validar_fruta(borde)


def test_prediccion_de_fruta_no_se_rechaza(rutas):
    r = Predictor(*rutas, usar_rembg=False).predecir(imagen_sintetica(), "clasico")
    assert r.rechazo is None
