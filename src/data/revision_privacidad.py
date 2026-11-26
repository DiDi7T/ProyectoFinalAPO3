"""
Revision de privacidad de las fotografias recolectadas.

Para que
--------
Las fotografias propias se tomaron en supermercados, plazas de mercado y
hogares. Antes de publicarlas en el repositorio o de compartirlas con los
demas grupos del curso hay que verificar que no contengan personas
identificables: mientras el material vive en el computador de quien lo tomo no
hay problema, pero compartirlo distribuye la imagen de terceros que nunca
dieron su consentimiento.

Corresponde al dilema 1 de `docs/03_etica_e_impactos.md` y al criterio 1 de la
rubrica. La diferencia entre escribir "somos conscientes de la privacidad" y
"revisamos las 1.019 fotografias con un detector de rostros y tratamos las N
marcadas" es justamente lo que la rubrica evalua.

Que hace y que NO hace
----------------------
Detecta **rostros** con YuNet, una red pequena de deteccion facial incluida en
OpenCV. Una mano sosteniendo fruta no se marca, que es el caso mayoritario del
material.

NO detecta letreros de locales, matriculas, documentos ni etiquetas de precio
con nombre del establecimiento. Esta es una **primera pasada automatica que
reduce el trabajo humano**, no un certificado: la revision final la hace una
persona sobre la hoja de contacto que genera este script.

El modelo se descarga una vez (~230 KB) desde el repositorio oficial de
OpenCV y se guarda en data_processed/modelos/.

Uso
---
    python -m src.data.revision_privacidad
    python -m src.data.revision_privacidad --umbral 0.5

Salidas
-------
    docs/anotacion/revision_privacidad.csv       una fila por imagen marcada
    docs/anotacion/revision_privacidad.jpg       hoja de contacto de lo marcado
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import DOCS, RECOLECTADO_DIR

MODELO = Path("data_processed/modelos/face_detection_yunet_2023mar.onnx")
URL_MODELO = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SALIDA_CSV = DOCS / "anotacion" / "revision_privacidad.csv"
SALIDA_HOJA = DOCS / "anotacion" / "revision_privacidad.jpg"

LADO_DETECCION = 640


def asegurar_modelo(ruta: Path = MODELO) -> Path:
    """Descarga el detector si no esta presente."""
    if ruta.exists():
        return ruta

    import urllib.request

    ruta.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando el detector de rostros desde {URL_MODELO}")
    urllib.request.urlretrieve(URL_MODELO, ruta)
    print(f"Guardado en {ruta} ({ruta.stat().st_size / 1024:.0f} KB)")
    return ruta


def detectar_rostros(detector, imagen: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    """
    Devuelve las cajas (x, y, w, h, confianza) de los rostros detectados.

    La imagen se reduce a LADO_DETECCION en su lado mayor antes de analizarla:
    acelera el barrido y no pierde rostros, porque un rostro demasiado pequeno
    para verse a esa escala tampoco seria identificable.
    """
    alto, ancho = imagen.shape[:2]
    escala = LADO_DETECCION / max(alto, ancho)
    if escala < 1.0:
        imagen = cv2.resize(imagen, (int(ancho * escala), int(alto * escala)))
    else:
        escala = 1.0

    h, w = imagen.shape[:2]
    detector.setInputSize((w, h))
    _, caras = detector.detect(imagen)
    if caras is None:
        return []

    resultado = []
    for cara in caras:
        x, y, cw, ch = cara[:4]
        resultado.append(
            (int(x / escala), int(y / escala), int(cw / escala), int(ch / escala), float(cara[-1]))
        )
    return resultado


def revisar(
    carpeta: Path = RECOLECTADO_DIR,
    umbral: float = 0.6,
    columnas: int = 6,
) -> dict:
    """Recorre las fotografias, marca las que contienen rostros y escribe el informe."""
    ruta_modelo = asegurar_modelo()
    detector = cv2.FaceDetectorYN.create(
        str(ruta_modelo), "", (320, 320), umbral, 0.3, 5000
    )

    imagenes = sorted(carpeta.rglob("*.jpg"))
    if not imagenes:
        raise FileNotFoundError(f"No hay imagenes en {carpeta}")

    print(f"Revisando {len(imagenes)} fotografias con umbral {umbral}...")
    marcadas: list[dict] = []
    ilegibles = 0

    for i, ruta in enumerate(imagenes, start=1):
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            ilegibles += 1
            continue

        caras = detectar_rostros(detector, imagen)
        if caras:
            area_imagen = imagen.shape[0] * imagen.shape[1]
            for x, y, w, h, conf in caras:
                marcadas.append(
                    {
                        "filepath": ruta.relative_to(carpeta.parents[1]).as_posix(),
                        "confianza": round(conf, 3),
                        "x": x, "y": y, "ancho": w, "alto": h,
                        "fraccion_imagen": round(w * h / area_imagen, 4),
                    }
                )

        if i % 200 == 0:
            print(f"  ... {i}/{len(imagenes)}  marcadas hasta ahora: {len(marcadas)}")

    SALIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        campos = ["filepath", "confianza", "x", "y", "ancho", "alto", "fraccion_imagen"]
        escritor = csv.DictWriter(fh, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(marcadas)

    # Hoja de contacto con los recortes marcados, para la revision humana.
    rutas_unicas = sorted({m["filepath"] for m in marcadas})
    if rutas_unicas:
        celdas = []
        for rel in rutas_unicas:
            img = cv2.imread(str(carpeta.parents[1] / rel))
            if img is None:
                continue
            for m in marcadas:
                if m["filepath"] != rel:
                    continue
                cv2.rectangle(
                    img, (m["x"], m["y"]),
                    (m["x"] + m["ancho"], m["y"] + m["alto"]), (0, 0, 255), 4
                )
            celda = cv2.resize(img, (260, 260))
            banda = np.full((30, 260, 3), 255, dtype=np.uint8)
            cv2.putText(banda, Path(rel).name[:30], (5, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            celdas.append(np.vstack([banda, celda]))

        while len(celdas) % columnas:
            celdas.append(np.full((290, 260, 3), 245, dtype=np.uint8))
        filas = [np.hstack(celdas[i : i + columnas]) for i in range(0, len(celdas), columnas)]
        rejilla = np.vstack(filas)
        cab = np.full((44, rejilla.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(cab, "Posibles rostros detectados - revisar una por una",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(str(SALIDA_HOJA), np.vstack([cab, rejilla]))

    resumen = {
        "n_revisadas": len(imagenes) - ilegibles,
        "n_ilegibles": ilegibles,
        "n_imagenes_marcadas": len(rutas_unicas),
        "n_detecciones": len(marcadas),
        "umbral": umbral,
    }

    print("\n=== RESUMEN DE LA REVISION ===")
    print(f"Fotografias revisadas:  {resumen['n_revisadas']}")
    print(f"Imagenes marcadas:      {resumen['n_imagenes_marcadas']}")
    print(f"Detecciones totales:    {resumen['n_detecciones']}")
    if rutas_unicas:
        print(f"\nInforme: {SALIDA_CSV}")
        print(f"Hoja de contacto para revisar a ojo: {SALIDA_HOJA}")
        print(
            "\nCada imagen marcada debe revisarse a ojo: el detector produce falsos\n"
            "positivos sobre texturas y manos. Para las que si tengan una persona,\n"
            "recortar o difuminar la region antes de compartir el dataset."
        )
    else:
        print("\nNinguna imagen contiene rostros detectables.")
    print(
        "\nRecordatorio: esta pasada NO detecta letreros, documentos ni etiquetas\n"
        "con el nombre del establecimiento. Eso sigue siendo revision humana."
    )
    return resumen


def main() -> None:
    from src.utils.registro import registrar_ejecucion

    with registrar_ejecucion("revision_privacidad"):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--umbral",
            type=float,
            default=0.6,
            help="Confianza minima para marcar un rostro (bajarlo marca mas)",
        )
        args = parser.parse_args()
        revisar(umbral=args.umbral)


if __name__ == "__main__":
    main()
