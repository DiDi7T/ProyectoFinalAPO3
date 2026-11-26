
"""
Segmentación de fotografías recolectadas mediante rembg.

Lee labels.csv y guarda las máscaras binarias,
las frutas sobre fondo negro y los recortes 224x224.

No modifica las fotografías ni las etiquetas originales.
"""

import argparse
import csv

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.data.build_dataset import read_image


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABELS = PROJECT_ROOT / "data_processed" / "labels.csv"

SALIDA = (
    PROJECT_ROOT
    / "data_processed"
    / "rembg_recolectado"
)

REPORTE = (
    PROJECT_ROOT
    / "data_processed"
    / "rembg_recolectado_manifest.csv"
)

UMBRAL_ALPHA = 127
TAMANO = 224


# ============================================================
# 2. REDIMENSIONAR Y CENTRAR LA FRUTA
# ============================================================

def encuadrar_fruta(imagen, mascara):
    """
    Recorta la región detectada y la ajusta a
    224x224 conservando su relación de aspecto.
    """

    ys, xs = np.where(mascara > 0)

    if len(xs) == 0:
        raise ValueError("Máscara vacía")

    alto, ancho = imagen.shape[:2]

    margen = max(
        2,
        int(
            0.03 * max(
                xs.max() - xs.min() + 1,
                ys.max() - ys.min() + 1
            )
        )
    )

    x0 = max(0, int(xs.min()) - margen)
    x1 = min(ancho, int(xs.max()) + margen + 1)

    y0 = max(0, int(ys.min()) - margen)
    y1 = min(alto, int(ys.max()) + margen + 1)

    recorte = imagen[y0:y1, x0:x1]

    h, w = recorte.shape[:2]

    escala = TAMANO / max(h, w)

    nuevo_ancho = max(1, round(w * escala))
    nuevo_alto = max(1, round(h * escala))

    interpolacion = (
        cv2.INTER_AREA
        if escala < 1
        else cv2.INTER_LINEAR
    )

    redimensionada = cv2.resize(
        recorte,
        (nuevo_ancho, nuevo_alto),
        interpolation=interpolacion
    )

    resultado = np.zeros(
        (TAMANO, TAMANO, 3),
        dtype=np.uint8
    )

    x = (TAMANO - nuevo_ancho) // 2
    y = (TAMANO - nuevo_alto) // 2

    resultado[
        y:y + nuevo_alto,
        x:x + nuevo_ancho
    ] = redimensionada

    return resultado


# ============================================================
# 3. CARGAR LAS FOTOGRAFÍAS RECOLECTADAS
# ============================================================

def cargar_fotografias():

    if not LABELS.is_file():
        raise FileNotFoundError(
            f"No existe: {LABELS}"
        )

    with LABELS.open(
        newline="",
        encoding="utf-8-sig"
    ) as archivo:

        filas = [
            fila
            for fila in csv.DictReader(archivo)
            if fila.get("source", "").lower().strip()
            == "recolectado"
        ]

    if not filas:
        raise RuntimeError(
            "No se encontraron imágenes recolectadas."
        )

    return sorted(
        filas,
        key=lambda fila: fila["filepath"]
    )


# ============================================================
# 4. SELECCIONAR IMÁGENES PARA LA PRUEBA
# ============================================================

def seleccionar_fotografias(filas, limite):

    if limite is None:
        return filas

    # Agrupar por fruta y calidad para que la muestra
    # contenga distintas categorías.
    grupos = defaultdict(list)

    for fila in filas:

        clave = (
            fila["fruit"],
            fila["quality"]
        )

        grupos[clave].append(fila)

    seleccion = []

    while (
        len(seleccion) < limite
        and any(grupos.values())
    ):

        for clave in sorted(grupos):

            if (
                grupos[clave]
                and len(seleccion) < limite
            ):
                seleccion.append(
                    grupos[clave].pop(0)
                )

    return seleccion


# ============================================================
# 5. PROCESAMIENTO DEL DATASET
# ============================================================

def procesar_dataset(limite=None):

    filas = cargar_fotografias()

    filas = seleccionar_fotografias(
        filas,
        limite
    )

    print("=" * 60)
    print("SEGMENTACIÓN REMBG - APO 3")
    print("=" * 60)

    print("Fotografías seleccionadas:", len(filas))

    registros = []

    sesion = None

    carpeta_origen = (
        PROJECT_ROOT / "data_raw" / "recolectado"
    ).resolve()

    for numero, fila in enumerate(filas, start=1):

        ruta_relativa = Path(fila["filepath"])

        registro = {
            "filepath": ruta_relativa.as_posix(),
            "fruit": fila["fruit"],
            "quality": fila["quality"],
            "mask_filepath": "",
            "isolated_filepath": "",
            "cnn_filepath": "",
            "relative_diameter_rembg": "",
            "size_label_rembg": "",
            "mask_fraction": "",
            "status": "error"
        }

        try:

            ruta_original = (
                PROJECT_ROOT / ruta_relativa
            ).resolve()

            # Solo procesar archivos de recolectado.
            subruta = ruta_original.relative_to(
                carpeta_origen
            )

            imagen = read_image(
                ruta_original,
                reduced_jpeg=False
            )

            if imagen is None:
                raise ValueError(
                    "No se pudo leer la fotografía"
                )

            carpeta_destino = (
                SALIDA / subruta.parent
            )

            carpeta_destino.mkdir(
                parents=True,
                exist_ok=True
            )

            nombre = (
                f"{subruta.stem}_"
                f"{subruta.suffix.lower().lstrip('.')}"
            )

            ruta_mascara = (
                carpeta_destino
                / f"{nombre}_mask.png"
            )

            ruta_aislada = (
                carpeta_destino
                / f"{nombre}_isolated.png"
            )

            ruta_cnn = (
                carpeta_destino
                / f"{nombre}_224.png"
            )

            # ------------------------------------------------
            # A. GENERAR O REUTILIZAR LA SEGMENTACIÓN
            # ------------------------------------------------

            archivos_existen = all(
                ruta.is_file()
                for ruta in [
                    ruta_mascara,
                    ruta_aislada,
                    ruta_cnn
                ]
            )

            if archivos_existen:

                mascara = cv2.imread(
                    str(ruta_mascara),
                    cv2.IMREAD_GRAYSCALE
                )

                if (
                    mascara is None
                    or mascara.shape != imagen.shape[:2]
                ):
                    raise ValueError(
                        "Máscara guardada incompatible"
                    )

            else:

                # Cargar el modelo una sola vez.
                if sesion is None:

                    from rembg import new_session

                    sesion = new_session()

                from rembg import remove

                imagen_rgb = cv2.cvtColor(
                    imagen,
                    cv2.COLOR_BGR2RGB
                )

                salida_rgba = remove(
                    Image.fromarray(imagen_rgb),
                    session=sesion
                ).convert("RGBA")

                rgba = np.asarray(salida_rgba)

                alpha = rgba[:, :, 3]

                mascara = np.where(
                    alpha >= UMBRAL_ALPHA,
                    255,
                    0
                ).astype(np.uint8)

                if not np.any(mascara):
                    raise ValueError(
                        "La máscara está vacía"
                    )

                # Conservar la fruta a color.
                aislada = cv2.bitwise_and(
                    imagen,
                    imagen,
                    mask=mascara
                )

                # Recorte final de 224x224.
                recorte = encuadrar_fruta(
                    aislada,
                    mascara
                )

                # Guardar los tres resultados.
                resultados = [
                    (ruta_mascara, mascara),
                    (ruta_aislada, aislada),
                    (ruta_cnn, recorte)
                ]

                for ruta, resultado in resultados:

                    if not cv2.imwrite(
                        str(ruta),
                        resultado
                    ):
                        raise OSError(
                            f"No se pudo guardar {ruta}"
                        )

            # ------------------------------------------------
            # B. CARACTERÍSTICAS DE LA MÁSCARA
            # ------------------------------------------------

            proporcion = (
                float(np.count_nonzero(mascara))
                / mascara.size
            )

            contornos, _ = cv2.findContours(
                mascara,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            area = max(
                (
                    cv2.contourArea(c)
                    for c in contornos
                ),
                default=0
            )

            if area <= 0:
                raise ValueError(
                    "No se detectó una región válida"
                )

            diametro = np.sqrt(
                4 * area / np.pi
            )

            diametro_relativo = float(
                diametro / max(mascara.shape)
            )

            # Umbrales actuales del proyecto de Diana.
            if diametro_relativo < 0.35:
                tamano = "pequeño"

            elif diametro_relativo <= 0.58:
                tamano = "mediano"

            else:
                tamano = "grande"

            estado = (
                "revisar"
                if proporcion < 0.01
                or proporcion > 0.95
                else "generada"
            )

            registro.update({
                "mask_filepath": (
                    ruta_mascara.relative_to(
                        PROJECT_ROOT
                    ).as_posix()
                ),
                "isolated_filepath": (
                    ruta_aislada.relative_to(
                        PROJECT_ROOT
                    ).as_posix()
                ),
                "cnn_filepath": (
                    ruta_cnn.relative_to(
                        PROJECT_ROOT
                    ).as_posix()
                ),
                "relative_diameter_rembg": round(
                    diametro_relativo, 4
                ),
                "size_label_rembg": tamano,
                "mask_fraction": round(
                    proporcion, 4
                ),
                "status": estado
            })

        except Exception as error:

            registro["status"] = (
                f"ERROR: {error}"
            )

        registros.append(registro)

        print(
            f"[{numero}/{len(filas)}] "
            f"{registro['fruit']} / "
            f"{registro['quality']}: "
            f"{registro['status']}"
        )

    # ========================================================
    # 6. GUARDAR REPORTE
    # ========================================================

    REPORTE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    columnas = [
        "filepath",
        "fruit",
        "quality",
        "mask_filepath",
        "isolated_filepath",
        "cnn_filepath",
        "relative_diameter_rembg",
        "size_label_rembg",
        "mask_fraction",
        "status"
    ]

    with REPORTE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=columnas
        )

        escritor.writeheader()
        escritor.writerows(registros)

    print("\n" + "=" * 60)
    print("SEGMENTACIÓN COMPLETADA")
    print("=" * 60)

    print("Imágenes evaluadas:", len(registros))
    print("Reporte:", REPORTE)
    print("Imágenes procesadas:", SALIDA)

    print(
        "\nLos datos originales y "
        "labels.csv no fueron modificados."
    )


# ============================================================
# 7. EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    grupo = parser.add_mutually_exclusive_group()

    grupo.add_argument(
        "--limit",
        type=int,
        default=18
    )

    grupo.add_argument(
        "--all",
        action="store_true"
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit debe ser positivo")

    procesar_dataset(
        limite=None if args.all else args.limit
    )
