"""
Interfaz grafica del clasificador de calidad de frutas.

Ejecutar desde la raiz del repositorio:

    streamlit run src/deployment/app.py

Permite subir una foto o tomarla con la camara, y muestra la calidad, el
tamano, las probabilidades y la region que el sistema tomo como fruta. Si la
confianza es baja lo dice, y el operario puede corregir la prediccion; las
correcciones quedan en experiments/logs/correcciones.csv.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

# Streamlit agrega al path la carpeta del script, no la raiz del proyecto.
RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.deployment.predict import UMBRAL_CONFIANZA, Predictor, leer_imagen  # noqa: E402
from src.utils.config import LOGS, QUALITY_CLASSES  # noqa: E402

NOMBRE_CALIDAD = {"good": "Buena", "regular": "Regular", "bad": "Mala"}
NOMBRE_TAMANO = {"pequeño": "Pequeño", "mediano": "Mediano", "grande": "Grande"}
# Demostrativo y nombre de cada fruta, para la frase "Este banano tiene ...".
FRASE_FRUTA = {
    "apple": "Esta manzana",
    "banana": "Este banano",
    "guava": "Esta guayaba",
    "lime": "Este limón",
    "orange": "Esta naranja",
}
NOMBRE_MODELO = {"cnn": "CNN", "clasico": "Modelo clásico"}
RUTA_CORRECCIONES = LOGS / "correcciones.csv"


@st.cache_resource
def cargar_predictor() -> Predictor:
    predictor = Predictor()
    predictor.calentar()
    return predictor


def contorno_sobre_imagen(imagen_bgr, mascara):
    """Dibuja el borde de la mascara sobre la foto, para ver que se analizo."""
    salida = imagen_bgr.copy()
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grosor = max(2, round(max(salida.shape[:2]) / 250))
    cv2.drawContours(salida, contornos, -1, (0, 255, 0), grosor)
    return cv2.cvtColor(salida, cv2.COLOR_BGR2RGB)


def guardar_correccion(nombre: str, resultado, correccion: str) -> None:
    RUTA_CORRECCIONES.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not RUTA_CORRECCIONES.exists()
    with open(RUTA_CORRECCIONES, "a", newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        if nuevo:
            escritor.writerow(["fecha", "archivo", "modelo", "prediccion", "confianza", "correccion"])
        escritor.writerow([
            datetime.now().isoformat(timespec="seconds"),
            nombre,
            resultado.modelo,
            resultado.calidad,
            f"{resultado.confianza_calidad:.4f}",
            correccion,
        ])


def tabla_probabilidades(proba: dict[str, float], nombres: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Clase": [nombres[c] for c in proba], "Probabilidad": [f"{p:.0%}" for p in proba.values()]}
    )


# ============================================================

st.set_page_config(page_title="Calidad de frutas", layout="centered")
st.title("Clasificación de calidad de frutas")
st.write(
    "Suba o tome una foto de **una sola fruta** sobre un fondo liso. "
    "Frutas soportadas: manzana, banano, guayaba, limón y naranja."
)

predictor = cargar_predictor()
if not predictor.disponibles:
    st.error(
        "No se encontraron modelos entrenados en `experiments/checkpoints/`. "
        "Entrene primero (`python -m src.training.train_classic --task both` "
        "y/o `python -m src.training.train_cnn`)."
    )
    st.stop()

col_origen, col_modelo = st.columns(2)
origen = col_origen.radio("Imagen", ["Subir archivo", "Cámara"], horizontal=True)
modelo = col_modelo.selectbox(
    "Modelo", predictor.disponibles, format_func=lambda m: NOMBRE_MODELO[m]
)

if origen == "Subir archivo":
    archivo = st.file_uploader(
        "Foto", type=["jpg", "jpeg", "png", "bmp", "webp", "heic", "heif"], label_visibility="collapsed"
    )
else:
    archivo = st.camera_input("Foto", label_visibility="collapsed")

if archivo is None:
    st.stop()

try:
    imagen = leer_imagen(archivo.getvalue())
except Exception as error:  # noqa: BLE001 - cualquier archivo ilegible
    st.error(f"No se pudo leer la imagen: {error}")
    st.stop()

resultado = predictor.predecir(imagen, modelo)

st.divider()

if resultado.rechazo:
    st.image(contorno_sobre_imagen(imagen, resultado.mascara), caption="Región detectada", width="stretch")
    st.error(
        f"No se detectó una fruta válida: {resultado.rechazo}. "
        "Tome la foto con una sola fruta centrada, completa y sobre un fondo liso."
    )
    st.stop()

sujeto = FRASE_FRUTA.get(resultado.fruta, "Esta fruta")
st.subheader(f"{sujeto} tiene calidad {NOMBRE_CALIDAD[resultado.calidad].lower()}")

col_img, col_res = st.columns([3, 2])
col_img.image(contorno_sobre_imagen(imagen, resultado.mascara), caption="Región analizada", width="stretch")

with col_res:
    st.metric("Calidad", NOMBRE_CALIDAD[resultado.calidad])
    st.caption(f"Confianza: {resultado.confianza_calidad:.0%}")
    st.metric("Tamaño", NOMBRE_TAMANO[resultado.tamano])
    st.caption(f"Diámetro relativo: {resultado.diametro_relativo:.2f}")
    st.caption(f"{NOMBRE_MODELO[resultado.modelo]} · {resultado.tiempo_ms:.0f} ms")

if resultado.baja_confianza:
    st.warning(
        f"Confianza baja (menos de {UMBRAL_CONFIANZA:.0%}). "
        "Revise la fruta manualmente antes de clasificarla."
    )

with st.expander("Probabilidades"):
    c1, c2 = st.columns(2)
    c1.write("Calidad")
    c1.dataframe(tabla_probabilidades(resultado.proba_calidad, NOMBRE_CALIDAD), hide_index=True)
    c2.write("Tamaño")
    c2.dataframe(tabla_probabilidades(resultado.proba_tamano, NOMBRE_TAMANO), hide_index=True)
    st.caption(
        "El tamaño es relativo al encuadre: depende de la distancia de la cámara "
        "y no equivale a una medida física."
    )

with st.expander("Corregir la predicción"):
    with st.form("correccion", clear_on_submit=False):
        correccion = st.selectbox(
            "Calidad correcta",
            QUALITY_CLASSES,
            index=QUALITY_CLASSES.index(resultado.calidad),
            format_func=lambda c: NOMBRE_CALIDAD[c],
        )
        if st.form_submit_button("Guardar corrección"):
            guardar_correccion(getattr(archivo, "name", "camara"), resultado, correccion)
            st.success("Corrección guardada.")
