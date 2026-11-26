"""
Ingesta y normalizacion de las fotografias recolectadas por el grupo.

El material propio llega tal como lo produjeron los celulares de los
integrantes, y eso trae cuatro problemas que hay que resolver antes de
cualquier modelado:

1. Formatos heterogeneos. 425 de las 880 fotografias son HEIC (formato nativo
   de iPhone), que OpenCV no sabe leer: cv2.imread devuelve None y la imagen
   se perderia en silencio. Se convierten a JPEG con pillow-heif.
2. Orientacion EXIF. Los celulares guardan la foto en el sensor y anotan la
   rotacion en una etiqueta EXIF. PIL la respeta, OpenCV la ignora, asi que
   una foto vertical entra acostada al modelo. Se aplica exif_transpose y se
   escribe la imagen ya rotada, sin depender de la etiqueta.
3. Resolucion excesiva. 1,4 GB para 880 fotos (1,6 MB de promedio) a 12 MP.
   Como la CNN trabaja a 128x128 y los descriptores se calculan sobre la
   mascara, se redimensiona el lado mayor a LADO_MAXIMO pixeles. Se preserva
   la relacion de aspecto, de modo que ningun descriptor de forma cambia.
4. Nomenclatura inconsistente. Hay carpetas 'Apple_good' y 'apple_regular',
   un 'Orage_Bad' con el nombre mal escrito, y 26 fotografias sueltas en la
   raiz de las carpetas de calidad cuyo nombre de archivo es la unica fuente
   de la etiqueta (grupoAPO_<fruta>_<calidad>_NN.jpg, en espanol). Todo se
   normaliza a <Calidad>_Quality_Fruits/<fruta>_<calidad>/ en ingles, que es
   la estructura que espera src.data.build_dataset.

Uso
---
    python -m src.data.ingest_recolectado --zip "C:/ruta/proyectoAPO.zip"
    python -m src.data.ingest_recolectado --zip ... --lado-maximo 1280
    python -m src.data.ingest_recolectado --zip ... --limite-por-clase 50

Salida
------
    data_raw/recolectado/<Calidad>_Quality_Fruits/<fruta>_<calidad>/*.jpg
    data_raw/recolectado/MANIFEST_ingesta.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.utils.config import FRUIT_CLASSES, QUALITY_CLASSES, RANDOM_STATE, RECOLECTADO_DIR

try:  # Soporte HEIC: se registra el decodificador dentro de PIL
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_DISPONIBLE = True
except ModuleNotFoundError:  # pragma: no cover
    HEIC_DISPONIBLE = False

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
LADO_MAXIMO = 1024
CALIDAD_JPEG = 92

# Resolucion minima aceptable. Una imagen cuyo lado mayor es menor que esto no
# aporta informacion util: al reducirla a 128x128 para la CNN la fruta queda en
# unas pocas decenas de pixeles y los descriptores de textura (LBP) y de
# defecto se calculan sobre ruido de compresion. En el material recibido habia
# 28 archivos asi, todos recortes de solo dos bananos, el mas pequeno de
# 19x54 pixeles.
LADO_MINIMO = 256

# --- Normalizacion de nombres -------------------------------------------------
# Las claves se buscan como subcadena del nombre de la carpeta o del archivo, ya
# en minusculas. El orden importa: 'lima' antes que 'lime' no cambia nada, pero
# 'granadilla' debe ir antes que 'granada' si alguna vez se agrega.
FRUTAS = {
    "apple": "apple",
    "manzana": "apple",
    "banana": "banana",
    "banano": "banana",
    "platano": "banana",
    "guava": "guava",
    "guayaba": "guava",
    "lime": "lime",
    "lima": "lime",
    "limon": "lime",
    "orange": "orange",
    "orage": "orange",  # error de escritura presente en el material original
    "naranja": "orange",
    "granadilla": "granadilla",  # fuera del alcance de 5 frutas, se registra igual
}

CALIDADES = {
    "good": "good",
    "buena": "good",
    "bueno": "good",
    "regular": "regular",
    "bad": "bad",
    "mala": "bad",
    "malo": "bad",
    "podrida": "bad",
}

CARPETA_CALIDAD = {
    "good": "Good_Quality_Fruits",
    "regular": "Regular_Quality_Fruits",
    "bad": "Bad_Quality_Fruits",
}

EN_ALCANCE = set(FRUIT_CLASSES)


def _buscar(texto: str, tabla: dict[str, str]) -> str | None:
    """Primera coincidencia por subcadena, con la clave mas larga primero."""
    limpio = texto.lower()
    for clave in sorted(tabla, key=len, reverse=True):
        if clave in limpio:
            return tabla[clave]
    return None


def etiquetar_entrada(ruta_zip_interna: str) -> tuple[str | None, str | None]:
    """
    Deduce (fruta, calidad) de una entrada del zip.

    La carpeta de calidad manda sobre el nombre del archivo, porque es la
    decision explicita que tomo quien organizo el material. La fruta se busca
    primero en la subcarpeta y, si no hay, en el nombre del archivo: asi se
    recuperan las 26 fotografias sueltas en la raiz.
    """
    partes = ruta_zip_interna.split("/")
    nombre_archivo = partes[-1]

    calidad = None
    for parte in partes[:-1]:
        calidad = _buscar(parte, CALIDADES)
        if calidad:
            break
    if calidad is None:
        calidad = _buscar(nombre_archivo, CALIDADES)

    fruta = None
    for parte in reversed(partes[:-1]):
        fruta = _buscar(parte, FRUTAS)
        if fruta:
            break
    if fruta is None:
        fruta = _buscar(nombre_archivo, FRUTAS)

    return fruta, calidad


def normalizar_imagen(datos: bytes, lado_maximo: int = LADO_MAXIMO) -> Image.Image:
    """
    Decodifica, corrige la orientacion EXIF, pasa a RGB y redimensiona.

    El factor de escala es s = lado_maximo / max(ancho, alto), aplicado solo si
    s < 1. Al ser isotropico, todos los descriptores adimensionales (razon de
    aspecto, solidez, circularidad, diametro relativo) quedan invariantes.
    """
    img = Image.open(io.BytesIO(datos))
    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        # Las PNG traen canal alfa; se compone sobre blanco para no dejar
        # pixeles negros espureos que la segmentacion leeria como fruta.
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            fondo = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(fondo, img)
        img = img.convert("RGB")

    lado = max(img.size)
    if lado > lado_maximo:
        escala = lado_maximo / lado
        nuevo = (max(1, round(img.width * escala)), max(1, round(img.height * escala)))
        img = img.resize(nuevo, Image.LANCZOS)

    return img


def ingerir(
    ruta_zip: Path,
    destino: Path = RECOLECTADO_DIR,
    lado_maximo: int = LADO_MAXIMO,
    lado_minimo: int = LADO_MINIMO,
    limite_por_clase: int | None = None,
    semilla: int = RANDOM_STATE,
) -> dict:
    """Extrae, normaliza y reorganiza el material propio. Devuelve un resumen."""
    if not ruta_zip.exists():
        raise FileNotFoundError(f"No existe el zip: {ruta_zip}")

    destino.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semilla)

    with zipfile.ZipFile(ruta_zip) as zf:
        entradas = [
            i.filename
            for i in zf.infolist()
            if not i.is_dir() and Path(i.filename).suffix.lower() in EXTENSIONES
        ]
        print(f"Entradas de imagen en el zip: {len(entradas)}")

        # 1. Etiquetar todo antes de escribir, para poder aplicar cupos.
        clasificadas: dict[tuple[str, str], list[str]] = defaultdict(list)
        sin_etiqueta: list[str] = []
        for entrada in sorted(entradas):
            fruta, calidad = etiquetar_entrada(entrada)
            if fruta is None or calidad is None:
                sin_etiqueta.append(entrada)
                continue
            clasificadas[(fruta, calidad)].append(entrada)

        if sin_etiqueta:
            print(f"[AVISO] {len(sin_etiqueta)} entradas sin etiqueta deducible:")
            for e in sin_etiqueta[:10]:
                print(f"         {e}")

        # 2. Escribir con conversion y normalizacion.
        manifiesto: list[dict] = []
        contadores: Counter = Counter()
        fallos: list[tuple[str, str]] = []
        descartadas: list[tuple[str, str]] = []

        for (fruta, calidad), lista in sorted(clasificadas.items()):
            seleccion = sorted(lista)
            if limite_por_clase and len(seleccion) > limite_por_clase:
                idx = rng.choice(len(seleccion), size=limite_por_clase, replace=False)
                seleccion = [seleccion[i] for i in sorted(idx)]

            carpeta = destino / CARPETA_CALIDAD[calidad] / f"{fruta}_{calidad}"
            carpeta.mkdir(parents=True, exist_ok=True)

            for entrada in seleccion:
                origen = Path(entrada)
                es_heic = origen.suffix.lower() in {".heic", ".heif"}
                if es_heic and not HEIC_DISPONIBLE:
                    fallos.append((entrada, "HEIC sin pillow-heif instalado"))
                    continue

                # Hay archivos con el mismo nombre base en la raiz de la
                # carpeta de calidad y dentro de la subcarpeta de la fruta
                # (p. ej. grupoAPO_guayaba_regular_01). Sin este desambiguado
                # el segundo sobreescribiria al primero en silencio.
                salida = carpeta / f"{origen.stem}.jpg"
                contador_dup = 1
                while salida.exists():
                    salida = carpeta / f"{origen.stem}_dup{contador_dup}.jpg"
                    contador_dup += 1

                try:
                    img = normalizar_imagen(zf.read(entrada), lado_maximo)
                    if max(img.size) < lado_minimo:
                        descartadas.append(
                            (entrada, f"resolucion {img.width}x{img.height} < {lado_minimo}")
                        )
                        continue
                    img.save(salida, format="JPEG", quality=CALIDAD_JPEG, optimize=True)
                except Exception as exc:  # imagen corrupta o formato no soportado
                    fallos.append((entrada, f"{type(exc).__name__}: {exc}"))
                    continue

                contadores[(fruta, calidad)] += 1
                manifiesto.append(
                    {
                        "zip_entry": entrada,
                        "filepath": salida.relative_to(destino.parents[1]).as_posix(),
                        "fruit": fruta,
                        "quality": calidad,
                        "en_alcance": fruta in EN_ALCANCE,
                        "formato_original": origen.suffix.lower().lstrip("."),
                        "convertida_desde_heic": es_heic,
                        "ancho": img.width,
                        "alto": img.height,
                    }
                )

            print(f"[OK]   {fruta:11s} {calidad:8s} -> {contadores[(fruta, calidad)]:4d} imagenes")

    ruta_manifiesto = destino / "MANIFEST_ingesta.csv"
    if manifiesto:
        with open(ruta_manifiesto, "w", newline="", encoding="utf-8") as fh:
            escritor = csv.DictWriter(fh, fieldnames=list(manifiesto[0].keys()))
            escritor.writeheader()
            escritor.writerows(manifiesto)

    # 3. Resumen legible: cruce fruta x calidad y conteos de interes.
    print("\n=== RESUMEN DE INGESTA ===")
    print(f"{'fruta':12s}" + "".join(f"{c:>10s}" for c in QUALITY_CLASSES) + f"{'total':>10s}")
    frutas_vistas = sorted({f for f, _ in contadores})
    for fruta in frutas_vistas:
        fila = [contadores[(fruta, c)] for c in QUALITY_CLASSES]
        marca = "" if fruta in EN_ALCANCE else "  (fuera de alcance)"
        print(f"{fruta:12s}" + "".join(f"{v:>10d}" for v in fila) + f"{sum(fila):>10d}{marca}")
    totales = [sum(contadores[(f, c)] for f in frutas_vistas) for c in QUALITY_CLASSES]
    print(f"{'TOTAL':12s}" + "".join(f"{v:>10d}" for v in totales) + f"{sum(totales):>10d}")

    convertidas = sum(1 for m in manifiesto if m["convertida_desde_heic"])
    en_alcance = sum(1 for m in manifiesto if m["en_alcance"])
    print(f"\nImagenes escritas:            {len(manifiesto)}")
    print(f"  convertidas desde HEIC:     {convertidas}")
    print(f"  dentro de las 5 frutas:     {en_alcance}")
    print(f"  fuera de alcance:           {len(manifiesto) - en_alcance}")
    if fallos:
        print(f"\n[AVISO] {len(fallos)} imagenes no se pudieron procesar:")
        for entrada, motivo in fallos[:10]:
            print(f"         {entrada}: {motivo}")
    print(f"\nManifiesto: {ruta_manifiesto}")
    print(f"Destino:    {destino}")

    return {
        "n_escritas": len(manifiesto),
        "n_convertidas_heic": convertidas,
        "n_en_alcance": en_alcance,
        "n_fallos": len(fallos),
        "n_descartadas_resolucion": len(descartadas),
        "n_sin_etiqueta": len(sin_etiqueta),
        "conteos": {f"{f}_{c}": v for (f, c), v in contadores.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path, help="Zip con las fotografias propias")
    parser.add_argument("--lado-maximo", type=int, default=LADO_MAXIMO)
    parser.add_argument(
        "--lado-minimo",
        type=int,
        default=LADO_MINIMO,
        help="Descartar imagenes cuyo lado mayor sea menor que este valor",
    )
    parser.add_argument(
        "--limite-por-clase",
        type=int,
        default=None,
        help="Maximo de imagenes por combinacion fruta x calidad",
    )
    args = parser.parse_args()

    if not HEIC_DISPONIBLE:
        print("[AVISO] pillow-heif no esta instalado: las fotografias HEIC se omitiran.")
        print("        Instalalo con: pip install pillow-heif")

    ingerir(
        args.zip,
        lado_maximo=args.lado_maximo,
        lado_minimo=args.lado_minimo,
        limite_por_clase=args.limite_por_clase,
    )


if __name__ == "__main__":
    main()
