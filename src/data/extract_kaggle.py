"""
Extracción reproducible del dataset público 'Fruit Quality Classification'.

Fuente: https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification

El archivo descargado de Kaggle pesa ~3.2 GB y contiene 19.526 imágenes muy
desbalanceadas (por ejemplo 5.940 imágenes de 'Pomegranate_Good' frente a
~1.100 de la mayoría de clases). Para poder iterar en CPU sin perder
representatividad, este script extrae una MUESTRA ESTRATIFICADA: un número
fijo de imágenes por cada carpeta (fruta × calidad), elegidas con una semilla
fija para que todo el equipo obtenga exactamente el mismo subconjunto.

Uso:
    python -m src.data.extract_kaggle --zip "C:/ruta/fruits.zip"
    python -m src.data.extract_kaggle --zip ... --per-class 250 --mixed 40
    python -m src.data.extract_kaggle --zip ... --per-class 0   # todo el dataset

Salida:
    data_raw/kaggle/archive/Good Quality_Fruits/<Fruta>_Good/*.jpg
    data_raw/kaggle/archive/Bad Quality_Fruits/<Fruta>_Bad/*.jpg
    data_raw/kaggle/archive/Mixed Qualit_Fruits/<Fruta>/*.jpg
    data_raw/kaggle/MANIFEST_extraccion.csv   (trazabilidad de la muestra)
"""

from __future__ import annotations

import argparse
import csv
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from src.utils.config import FRUIT_CLASSES, KAGGLE_DIR, RANDOM_STATE

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Frutas dentro del alcance del proyecto. 'Pomegranate' se excluye porque no
# forma parte de las 5 clases definidas por el grupo y porque concentra el
# 30 % de las imágenes del dataset, lo que distorsionaría el balance.
FRUTAS_EN_ALCANCE = {f.lower() for f in FRUIT_CLASSES} | {"lemon"}


def _fruta_de_carpeta(nombre_carpeta: str) -> str | None:
    """Devuelve la fruta en alcance mencionada por el nombre de la carpeta."""
    limpio = nombre_carpeta.lower()
    for fruta in FRUTAS_EN_ALCANCE:
        if fruta in limpio:
            return fruta
    return None


def _agrupar_por_carpeta(nombres: list[str]) -> dict[str, list[str]]:
    """Agrupa las entradas del zip por carpeta contenedora."""
    grupos: dict[str, list[str]] = defaultdict(list)
    for nombre in nombres:
        if nombre.endswith("/"):
            continue
        if Path(nombre).suffix.lower() not in VALID_EXTENSIONS:
            continue
        grupos["/".join(nombre.split("/")[:-1])].append(nombre)
    return grupos


def seleccionar_muestra(
    grupos: dict[str, list[str]],
    por_clase: int,
    mixtas: int,
    semilla: int = RANDOM_STATE,
) -> list[tuple[str, str]]:
    """
    Elige la muestra estratificada.

    Devuelve una lista de pares (entrada_en_zip, carpeta_destino_relativa).
    `por_clase = 0` significa "todas las imágenes de la carpeta".
    """
    rng = random.Random(semilla)
    seleccion: list[tuple[str, str]] = []

    for carpeta, archivos in sorted(grupos.items()):
        nombre_hoja = carpeta.split("/")[-1]
        if _fruta_de_carpeta(nombre_hoja) is None:
            print(f"[INFO] Fuera de alcance, se omite: {carpeta}")
            continue

        es_mixta = "mixed" in carpeta.lower()
        cupo = mixtas if es_mixta else por_clase

        archivos_ordenados = sorted(archivos)
        if cupo and len(archivos_ordenados) > cupo:
            elegidos = rng.sample(archivos_ordenados, cupo)
        else:
            elegidos = archivos_ordenados

        for entrada in sorted(elegidos):
            seleccion.append((entrada, carpeta))

        print(f"[OK]   {carpeta}: {len(elegidos)}/{len(archivos_ordenados)}")

    return seleccion


def extraer(ruta_zip: Path, por_clase: int, mixtas: int, destino: Path = KAGGLE_DIR) -> Path:
    """Extrae la muestra estratificada y escribe el manifiesto de trazabilidad."""
    if not ruta_zip.exists():
        raise FileNotFoundError(f"No existe el archivo zip: {ruta_zip}")

    destino.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ruta_zip) as zf:
        grupos = _agrupar_por_carpeta(zf.namelist())
        print(f"Carpetas detectadas en el zip: {len(grupos)}")

        seleccion = seleccionar_muestra(grupos, por_clase, mixtas)
        print(f"\nImágenes seleccionadas: {len(seleccion)}")

        manifiesto = []
        for i, (entrada, carpeta) in enumerate(seleccion, start=1):
            carpeta_destino = destino / carpeta
            carpeta_destino.mkdir(parents=True, exist_ok=True)

            nombre_archivo = Path(entrada).name
            ruta_final = carpeta_destino / nombre_archivo

            if not ruta_final.exists():
                with zf.open(entrada) as origen, open(ruta_final, "wb") as salida:
                    salida.write(origen.read())

            manifiesto.append(
                {
                    "zip_entry": entrada,
                    "filepath": ruta_final.relative_to(destino.parents[1]).as_posix(),
                    "carpeta": carpeta,
                }
            )

            if i % 250 == 0:
                print(f"  ... {i}/{len(seleccion)} imágenes extraídas")

    ruta_manifiesto = destino.parent / "MANIFEST_extraccion.csv"
    with open(ruta_manifiesto, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=["zip_entry", "filepath", "carpeta"])
        escritor.writeheader()
        escritor.writerows(manifiesto)

    print(f"\nManifiesto de extracción: {ruta_manifiesto}")
    print(f"Destino de las imágenes:  {destino}")
    return ruta_manifiesto


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path, help="Ruta al zip de Kaggle")
    parser.add_argument(
        "--per-class",
        type=int,
        default=250,
        help="Imágenes por carpeta fruta×calidad (0 = todas)",
    )
    parser.add_argument(
        "--mixed",
        type=int,
        default=40,
        help="Imágenes por carpeta de Mixed Quality, para anotación manual (0 = todas)",
    )
    args = parser.parse_args()

    extraer(args.zip, args.per_class, args.mixed)


if __name__ == "__main__":
    main()
