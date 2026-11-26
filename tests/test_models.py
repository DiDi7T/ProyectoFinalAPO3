"""
Pruebas automaticas del pipeline.

Se ejecutan con:

    pytest -q

Las pruebas unitarias no necesitan datos. Las de integracion marcadas con
`skipif` solo corren si ya existe el cache de `src.features.build_features`,
de modo que `pytest` pase igual en una maquina recien clonada.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.evaluation.metrics import calcular_metricas, clases_presentes, costo_asimetrico
from src.features.build_features import (
    a_cuadrado,
    agrupar_duplicados,
    clave_original,
    dhash,
)
from src.features.extractors import N_FEATURES, extract_features, feature_names
from src.utils.config import FEATURES_DIR, QUALITY_CLASSES, SIZE_CLASSES


# ============================================================
# Utilidades de las pruebas
# ============================================================

def imagen_sintetica(alto: int = 240, ancho: int = 320, semilla: int = 0) -> np.ndarray:
    """Fondo gris poco saturado con un circulo naranja centrado (una 'fruta')."""
    import cv2

    rng = np.random.default_rng(semilla)
    img = np.full((alto, ancho, 3), 200, dtype=np.uint8)
    img = np.clip(img + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    centro = (ancho // 2, alto // 2)
    radio = min(alto, ancho) // 4
    cv2.circle(img, centro, radio, (20, 120, 230), thickness=-1)  # BGR naranja
    return img


# ============================================================
# Caracteristicas
# ============================================================

def test_vector_de_caracteristicas_tiene_la_dimension_declarada():
    v = extract_features(imagen_sintetica())
    assert v.shape == (N_FEATURES,)
    assert len(feature_names()) == N_FEATURES


def test_caracteristicas_son_finitas_incluso_con_mascara_vacia():
    img = imagen_sintetica()
    mascara_vacia = np.zeros(img.shape[:2], dtype=np.uint8)
    v = extract_features(img, mascara_vacia)
    assert np.isfinite(v).all()
    assert v.shape == (N_FEATURES,)


def test_histogramas_de_color_suman_uno():
    """Los tres histogramas marginales son distribuciones de probabilidad."""
    v = extract_features(imagen_sintetica())
    assert v[0:32].sum() == pytest.approx(1.0, abs=1e-4)      # H
    assert v[32:48].sum() == pytest.approx(1.0, abs=1e-4)     # S
    assert v[48:64].sum() == pytest.approx(1.0, abs=1e-4)     # V


def test_lbp_es_un_histograma_normalizado():
    v = extract_features(imagen_sintetica())
    inicio = 32 + 16 + 16 + 9 + 6
    assert v[inicio : inicio + 10].sum() == pytest.approx(1.0, abs=1e-4)


def test_descriptores_de_forma_detectan_un_circulo():
    """Un circulo debe dar circularidad cercana a 1 y excentricidad cercana a 0."""
    v = extract_features(imagen_sintetica())
    inicio_forma = 32 + 16 + 16 + 9
    circularidad = v[inicio_forma + 4]
    excentricidad = v[inicio_forma + 5]
    assert 0.8 < circularidad <= 1.05
    assert excentricidad < 0.4


def test_caracteristicas_invariantes_a_la_escala_de_la_imagen():
    """
    El mismo circulo fotografiado con el doble de resolucion debe producir
    practicamente el mismo vector: los descriptores son adimensionales.
    """
    v1 = extract_features(imagen_sintetica(240, 320))
    v2 = extract_features(imagen_sintetica(480, 640))
    inicio_forma = 32 + 16 + 16 + 9
    assert v1[inicio_forma] == pytest.approx(v2[inicio_forma], abs=0.02)      # d_rel
    assert v1[inicio_forma + 4] == pytest.approx(v2[inicio_forma + 4], abs=0.05)  # circularidad


# ============================================================
# Relleno a cuadrado (letterbox)
# ============================================================

@pytest.mark.parametrize("forma", [(300, 400, 3), (400, 300, 3), (128, 128, 3), (50, 900, 3)])
def test_a_cuadrado_devuelve_un_lienzo_cuadrado(forma):
    salida = a_cuadrado(np.zeros(forma, dtype=np.uint8), 128)
    assert salida.shape == (128, 128, 3)


def test_a_cuadrado_conserva_la_relacion_de_aspecto():
    """
    Un rectangulo horizontal y su transpuesto deben ocupar la misma area util
    en el lienzo: el escalado es isotropico, no un aplastamiento.
    """
    horizontal = np.full((300, 400, 3), 255, dtype=np.uint8)
    vertical = np.full((400, 300, 3), 255, dtype=np.uint8)

    area_h = int((a_cuadrado(horizontal, 128, relleno=0) == 255).all(axis=2).sum())
    area_v = int((a_cuadrado(vertical, 128, relleno=0) == 255).all(axis=2).sum())
    assert area_h == pytest.approx(area_v, rel=0.05)


def test_a_cuadrado_rellena_con_el_valor_pedido():
    salida = a_cuadrado(np.zeros((100, 400, 3), dtype=np.uint8), 128, relleno=114)
    assert salida[0, 0, 0] == 114
    assert salida[64, 64, 0] == 0


# ============================================================
# Deteccion de casi duplicados
# ============================================================

def test_dhash_es_estable_para_la_misma_imagen():
    img = imagen_sintetica()
    assert dhash(img) == dhash(img.copy())


def test_dhash_tolera_un_cambio_de_escala():
    """Es el caso que importa: el mismo objeto reescalado sigue siendo el mismo."""
    import cv2

    img = imagen_sintetica(400, 400)
    pequena = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    distancia = bin(dhash(img) ^ dhash(pequena)).count("1")
    assert distancia <= 6


def test_agrupar_duplicados_une_huellas_identicas():
    h = dhash(imagen_sintetica())
    grupos = agrupar_duplicados([h, h, h ^ (1 << 40) ^ (1 << 41)])
    assert grupos[0] == grupos[1]


def test_agrupar_duplicados_respeta_las_claves_de_archivo():
    """
    Dos imagenes con huellas muy distintas pero provenientes de la misma foto
    original (sufijo _aug_) deben quedar en el mismo grupo.
    """
    # Las tres huellas estan a distancia de Hamming 32 entre si, muy por
    # encima del umbral, de modo que solo la clave de archivo puede unirlas.
    hashes = [0x0000000000000000, 0xFFFFFFFFFFFFFFFF, 0x0F0F0F0F0F0F0F0F]
    claves = ["carpeta/foto_a", "carpeta/foto_a", "carpeta/foto_b"]
    grupos = agrupar_duplicados(hashes, claves_extra=claves)
    assert grupos[0] == grupos[1]
    assert grupos[0] != grupos[2]


@pytest.mark.parametrize(
    "ruta, esperado",
    [
        ("d/lime_bad/20260602_070421_aug_005.jpg", "d/lime_bad/20260602_070421"),
        ("d/lime_bad/20260602_070421.jpg", "d/lime_bad/20260602_070421"),
        ("d/guava/foto_01_dup1.jpg", "d/guava/foto_01"),
    ],
)
def test_clave_original_quita_sufijos_de_aumentacion_y_duplicado(ruta, esperado):
    assert clave_original(ruta) == esperado


# ============================================================
# Metricas
# ============================================================

def test_prediccion_perfecta_da_f1_uno():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = calcular_metricas(y, y, QUALITY_CLASSES)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["balanced_accuracy"] == pytest.approx(1.0)
    assert m["cohen_kappa"] == pytest.approx(1.0)


def test_clases_presentes_ignora_las_ausentes():
    y = np.array([0, 0, 2, 2])  # sin 'regular'
    indices, nombres = clases_presentes(y, QUALITY_CLASSES)
    assert indices == [0, 2]
    assert nombres == ["good", "bad"]


def test_mae_ordinal_penaliza_mas_el_error_de_dos_niveles():
    y = np.array([0, 0])
    cercano = np.array([1, 0])   # pequeno -> mediano
    lejano = np.array([2, 0])    # pequeno -> grande
    m_cerca = calcular_metricas(y, cercano, SIZE_CLASSES, ordinal=True)
    m_lejos = calcular_metricas(y, lejano, SIZE_CLASSES, ordinal=True)
    assert m_lejos["mae_ordinal"] > m_cerca["mae_ordinal"]


def test_costo_asimetrico_cuenta_los_falsos_buenos():
    y_true = np.array([2, 2, 0, 0])      # bad, bad, good, good
    y_pred = np.array([0, 2, 2, 0])      # un falso 'good' y un falso 'bad'
    c = costo_asimetrico(y_true, y_pred, idx_good=0, idx_bad=2)
    assert c["n_falsos_buenos"] == 1
    assert c["n_falsos_malos"] == 1
    assert c["riesgo_empirico"] == pytest.approx((5.0 + 1.0) / 4)


# ============================================================
# CNN
# ============================================================

def test_la_cnn_produce_dos_salidas_con_la_forma_correcta():
    torch = pytest.importorskip("torch")
    from src.models.cnn import FruitQualityCNN

    modelo = FruitQualityCNN(n_clases_calidad=3, n_clases_tamano=3)
    x = torch.randn(2, 3, 128, 128)
    logits_q, logits_s = modelo(x)
    assert logits_q.shape == (2, 3)
    assert logits_s.shape == (2, 3)
    assert modelo.n_parametros() > 0


def test_pesos_de_clase_compensan_el_desbalance():
    pytest.importorskip("torch")
    from src.models.cnn import pesos_de_clase

    y = np.array([0] * 90 + [1] * 10)  # 90 % de una clase
    w = pesos_de_clase(y, 3)
    assert w[1] > w[0]                 # la minoritaria pesa mas
    assert w[2] == 0.0                 # la ausente no aporta gradiente
    assert float(w[0]) == pytest.approx(100 / (2 * 90), rel=1e-5)


def test_normalizar_lote_entrega_el_formato_de_pytorch():
    pytest.importorskip("torch")
    from src.models.cnn import normalizar_lote

    lote = np.zeros((4, 128, 128, 3), dtype=np.uint8)
    x = normalizar_lote(lote)
    assert x.shape == (4, 3, 128, 128)
    assert x.dtype == np.float32


# ============================================================
# Integracion: el cache existente no debe tener fuga entre particiones
# ============================================================

hay_cache = (FEATURES_DIR / "split.npy").exists() and (FEATURES_DIR / "grupo.npy").exists()


@pytest.mark.skipif(not hay_cache, reason="requiere ejecutar src.features.build_features")
def test_ningun_grupo_de_duplicados_cruza_dos_particiones():
    """
    Es la prueba que protege el resultado principal del proyecto: si un grupo
    de imagenes casi duplicadas aparece en train y en test a la vez, las
    metricas reportadas estan infladas.
    """
    split = np.load(FEATURES_DIR / "split.npy", allow_pickle=True).astype(str)
    grupo = np.load(FEATURES_DIR / "grupo.npy")

    cruzados = [
        int(g) for g in np.unique(grupo) if len(np.unique(split[grupo == g])) > 1
    ]
    assert not cruzados, f"{len(cruzados)} grupos aparecen en mas de una particion"


@pytest.mark.skipif(not hay_cache, reason="requiere ejecutar src.features.build_features")
def test_las_tres_particiones_existen_y_no_estan_vacias():
    split = np.load(FEATURES_DIR / "split.npy", allow_pickle=True).astype(str)
    for nombre in ("train", "val", "test"):
        assert (split == nombre).sum() > 0, f"la particion {nombre} esta vacia"
