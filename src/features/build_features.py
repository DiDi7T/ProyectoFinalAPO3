"""
Construccion y cacheo de las matrices de entrada de los modelos.

Responsabilidades
-----------------
1. Particionar labels.csv en entrenamiento / validacion / prueba SIN fuga de
   informacion entre imagenes casi duplicadas.
2. Calcular una sola vez las caracteristicas de los modelos clasicos y los
   tensores de la CNN, y guardarlos en data_processed/features/*.npy.

Por que agrupar duplicados
--------------------------
El dataset publico de Kaggle contiene multiples fotografias casi identicas
de la misma fruta (variaciones de recorte, rotacion y brillo). Si la
particion es puramente aleatoria, copias de la misma fruta caen a la vez en
entrenamiento y en prueba: el modelo memoriza en lugar de generalizar y la
exactitud reportada queda inflada. Es el error metodologico mas comun en los
trabajos publicados sobre este dataset.

Para evitarlo se calcula una huella perceptual (dHash de 64 bits) por imagen,
se agrupan las imagenes cuya distancia de Hamming es menor o igual a
UMBRAL_HAMMING mediante una estructura union-find, y la particion se hace por
GRUPO con StratifiedGroupKFold: todas las copias de una misma fruta caen
siempre en la misma particion.

Uso
---
    python -m src.features.build_features
    python -m src.features.build_features --features baseline
    python -m src.features.build_features --no-cnn --limit 300
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.config import (
    CNN_IMAGE_SIZE,
    FEATURES_DIR,
    LABELS_CSV,
    PROJECT_ROOT,
    QUALITY_TO_IDX,
    RANDOM_STATE,
    SIZE_TO_IDX,
    FRUIT_TO_IDX,
    ensure_dirs,
)

UMBRAL_HAMMING = 6
TAMANO_DHASH = 8


# ============================================================
# 1. HUELLA PERCEPTUAL Y AGRUPAMIENTO DE DUPLICADOS
# ============================================================

def dhash(image_bgr: np.ndarray, tamano: int = TAMANO_DHASH) -> int:
    """
    Huella perceptual por diferencias horizontales (dHash) de 64 bits.

    Se reduce la imagen en escala de grises a (tamano+1) x tamano y se compara
    cada pixel con su vecino de la derecha; el bit k vale 1 si el pixel de la
    izquierda es mas brillante. El resultado es robusto a cambios de escala,
    compresion JPEG y pequenas variaciones de brillo, que es exactamente el
    tipo de duplicado que trae el dataset.
    """
    gris = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    pequena = cv2.resize(gris, (tamano + 1, tamano), interpolation=cv2.INTER_AREA)
    diferencias = pequena[:, 1:] > pequena[:, :-1]
    bits = diferencias.flatten()
    valor = 0
    for i, bit in enumerate(bits):
        if bit:
            valor |= 1 << i
    return valor


class UnionFind:
    """Union-find con compresion de camino, para agrupar imagenes similares."""

    def __init__(self, n: int) -> None:
        self.padre = list(range(n))

    def find(self, x: int) -> int:
        while self.padre[x] != x:
            self.padre[x] = self.padre[self.padre[x]]
            x = self.padre[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.padre[rb] = ra


def clave_original(ruta: str) -> str:
    """
    Identificador de la fotografia ORIGINAL de la que proviene un archivo.

    Parte del material recibido ya venia aumentado: en el conjunto propio hay
    148 archivos con el sufijo '_aug_NNN' que son variantes (rotacion, espejo,
    cambio de brillo) de una misma foto. Los casos extremos son lime_regular,
    con 39 archivos generados a partir de solo 5 fotos originales, y lime_bad,
    con 30 a partir de 7.

    Esas variantes NO pueden quedar repartidas entre entrenamiento y prueba:
    serian la misma fruta a ambos lados. La huella dHash no siempre las detecta
    porque es sensible a la rotacion, de modo que aqui se usa el nombre del
    archivo como evidencia adicional: se elimina el sufijo de aumentacion y el
    de desambiguacion, y lo que queda identifica la foto original dentro de su
    carpeta.
    """
    p = Path(ruta)
    base = re.sub(r"_aug_\d+$", "", p.stem)
    base = re.sub(r"_dup\d+$", "", base)
    return f"{p.parent.as_posix()}/{base}"


def agrupar_duplicados(
    hashes: list[int],
    umbral: int = UMBRAL_HAMMING,
    claves_extra: list[str] | None = None,
) -> np.ndarray:
    """
    Devuelve un identificador de grupo por imagen.

    Comparar todos los pares seria O(n^2). Se usa un esquema LSH sencillo: dos
    huellas a distancia de Hamming <= 6 comparten, por el principio del
    palomar, al menos uno de sus cuatro bloques de 16 bits cuando el umbral se
    reparte entre bloques. Solo se comparan los pares que caen en el mismo
    cubo, lo que reduce el costo a casi lineal.
    """
    n = len(hashes)
    uf = UnionFind(n)

    cubos: dict[tuple[int, int], list[int]] = {}
    for idx, h in enumerate(hashes):
        for bloque in range(4):
            clave = (bloque, (h >> (16 * bloque)) & 0xFFFF)
            cubos.setdefault(clave, []).append(idx)

    for indices in cubos.values():
        if len(indices) < 2 or len(indices) > 400:
            # Cubos gigantescos (p. ej. imagenes casi planas) se omiten para
            # no degradar a O(n^2); sus duplicados reales caen en otro bloque.
            continue
        for i in range(len(indices)):
            hi = hashes[indices[i]]
            for j in range(i + 1, len(indices)):
                if bin(hi ^ hashes[indices[j]]).count("1") <= umbral:
                    uf.union(indices[i], indices[j])

    # Evidencia adicional: archivos que provienen de la misma foto original
    # (variantes '_aug_NNN') se unen aunque sus huellas difieran por rotacion.
    if claves_extra is not None:
        por_clave: dict[str, list[int]] = {}
        for idx, clave in enumerate(claves_extra):
            por_clave.setdefault(clave, []).append(idx)
        for indices in por_clave.values():
            for j in indices[1:]:
                uf.union(indices[0], j)

    raices = {}
    grupos = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = uf.find(i)
        if r not in raices:
            raices[r] = len(raices)
        grupos[i] = raices[r]
    return grupos


# ============================================================
# 2. PARTICION POR GRUPOS
# ============================================================

def particionar_por_grupos(
    df: pd.DataFrame,
    grupos: np.ndarray,
    semilla: int = RANDOM_STATE,
) -> np.ndarray:
    """
    Asigna a cada fila la etiqueta de particion: 'train', 'val' o 'test'.

    Se usa StratifiedGroupKFold, que mantiene aproximadamente la proporcion de
    clases respetando a la vez la restriccion de grupo. Con 7 pliegues el
    primero aporta ~14,3 % para prueba; sobre el resto se repite con 6
    pliegues para obtener validacion.

    La estratificacion se hace por fruta x calidad x FUENTE. Incluir la fuente
    es necesario porque el dataset mezcla imagenes publicas de Kaggle (fondo
    de estudio, iluminacion controlada) con fotografias propias de plaza de
    mercado (fondos reales, sombras, manos). Si una fuente quedara concentrada
    en una sola particion, la metrica de prueba mediria un dominio distinto al
    de entrenamiento y no seria interpretable.
    """
    if "source" in df.columns:
        estratos = (
            df["fruit"].astype(str)
            + "_"
            + df["quality"].astype(str)
            + "_"
            + df["source"].astype(str)
        ).to_numpy()
    else:
        estratos = (df["fruit"].astype(str) + "_" + df["quality"].astype(str)).to_numpy()
    indices = np.arange(len(df))

    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=semilla)
    resto_idx, test_idx = next(sgkf.split(indices, estratos, groups=grupos))

    sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=semilla)
    sub_train, sub_val = next(
        sgkf2.split(resto_idx, estratos[resto_idx], groups=grupos[resto_idx])
    )

    particion = np.empty(len(df), dtype=object)
    particion[test_idx] = "test"
    particion[resto_idx[sub_val]] = "val"
    particion[resto_idx[sub_train]] = "train"
    return particion


# ============================================================
# 3. CONSTRUCCION DE LAS MATRICES
# ============================================================

GRIS_RELLENO = 114  # valor neutro para el relleno, equidistante del negro y del blanco


def a_cuadrado(
    imagen: np.ndarray,
    lado: int,
    interpolacion: int = cv2.INTER_AREA,
    relleno: int = GRIS_RELLENO,
) -> np.ndarray:
    """
    Lleva una imagen a lado x lado SIN deformarla.

    Redimensionar directamente a 128x128 con cv2.resize aplasta la imagen: una
    foto vertical 3:4 y una horizontal 4:3 de la misma fruta producen formas
    distintas, y la red tendria que aprender a compensar la orientacion de la
    camara. Como el material propio trae 590 fotos verticales y 290
    horizontales, el efecto no es despreciable.

    La solucion estandar es el escalado isotropico con relleno (letterbox):

        s = lado / max(H, W)        ->  (H', W') = (round(sH), round(sW))

    y luego se centra ese rectangulo en un lienzo lado x lado. Al ser el mismo
    factor s en ambos ejes, la relacion de aspecto de la fruta se conserva.
    El relleno usa un gris neutro en vez de negro para no introducir regiones
    oscuras que los descriptores de defecto interpretarian como manchas.
    """
    alto, ancho = imagen.shape[:2]
    escala = lado / max(alto, ancho)
    nuevo_alto = max(1, int(round(alto * escala)))
    nuevo_ancho = max(1, int(round(ancho * escala)))
    redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto), interpolation=interpolacion)

    if imagen.ndim == 3:
        lienzo = np.full((lado, lado, imagen.shape[2]), relleno, dtype=imagen.dtype)
    else:
        lienzo = np.full((lado, lado), relleno, dtype=imagen.dtype)

    arriba = (lado - nuevo_alto) // 2
    izquierda = (lado - nuevo_ancho) // 2
    lienzo[arriba : arriba + nuevo_alto, izquierda : izquierda + nuevo_ancho] = redimensionada
    return lienzo


def _mascara_de_fila(fila: pd.Series, imagen: np.ndarray) -> np.ndarray:
    """
    Devuelve la mascara de la fruta: la de rembg si el manifiesto la registro,
    y si no la heuristica de saturacion + Otsu.
    """
    ruta_mascara = fila.get("mask_filepath", "")
    if isinstance(ruta_mascara, str) and ruta_mascara:
        ruta = PROJECT_ROOT / ruta_mascara
        if ruta.exists():
            mascara = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
            if mascara is not None:
                return mascara

    from src.data.build_dataset import get_fruit_mask_and_size

    mascara, _, _ = get_fruit_mask_and_size(imagen)
    return mascara


def limitar_fuente(
    df: pd.DataFrame,
    fuente: str,
    maximo_por_clase: int,
    semilla: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Recorta una fuente a `maximo_por_clase` imagenes por combinacion
    fruta x calidad, dejando intactas las demas fuentes.

    Sirve para controlar la composicion del dataset. El conjunto publico tiene
    unas 10.000 imagenes utiles y ninguna de clase 'regular'; si se usaran
    todas, esa clase (que solo aportan las fotografias propias) caeria por
    debajo del 2 % del total y el sistema de tres clases se volveria
    inentrenable. Limitar la fuente publica mantiene el problema de tres
    clases con una proporcion razonable.
    """
    if maximo_por_clase <= 0 or "source" not in df.columns:
        return df

    de_la_fuente = df["source"] == fuente
    if not de_la_fuente.any():
        return df

    conservar = [df.index[~de_la_fuente]]
    for _, grupo in df[de_la_fuente].groupby(["fruit", "quality"], observed=True):
        if len(grupo) > maximo_por_clase:
            grupo = grupo.sample(n=maximo_por_clase, random_state=semilla)
        conservar.append(grupo.index)

    indices = np.sort(np.concatenate([np.asarray(c) for c in conservar]))
    recortado = df.loc[indices].reset_index(drop=True)
    print(
        f"Fuente '{fuente}' limitada a {maximo_por_clase} por fruta x calidad: "
        f"{len(df)} -> {len(recortado)} imagenes"
    )
    return recortado


def construir(
    tipo_features: str = "enriched",
    con_cnn: bool = True,
    limite: int | None = None,
    max_kaggle: int = 0,
    semilla: int = RANDOM_STATE,
) -> dict:
    """
    Calcula huellas, particiones, caracteristicas y tensores, y los guarda.

    Returns
    -------
    dict con el resumen (conteos por particion y rutas de los .npy).
    """
    ensure_dirs()

    if not LABELS_CSV.exists():
        raise FileNotFoundError(
            f"No existe {LABELS_CSV}. Ejecuta primero: python -m src.data.build_dataset"
        )

    df = pd.read_csv(LABELS_CSV)
    df = limitar_fuente(df, "kaggle", max_kaggle, semilla)
    if limite:
        df = df.sample(n=min(limite, len(df)), random_state=semilla).reset_index(drop=True)

    print(f"Imagenes en labels.csv: {len(df)}")

    if tipo_features == "enriched":
        from src.features.extractors import extract_features, feature_names

        extractor = extract_features
        nombres = feature_names()
    elif tipo_features == "baseline":
        from src.data.preprocess import extract_ml_features

        def extractor(imagen, mask=None):  # noqa: ARG001 - firma homogenea
            return extract_ml_features(imagen)

        nombres = []
    else:
        raise ValueError("tipo_features debe ser 'enriched' o 'baseline'")

    filas_validas: list[int] = []
    hashes: list[int] = []
    lista_features: list[np.ndarray] = []
    lista_imgs: list[np.ndarray] = []
    lista_masks: list[np.ndarray] = []

    for i, fila in df.iterrows():
        ruta = PROJECT_ROOT / fila["filepath"]
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            print(f"[AVISO] ilegible, se descarta: {ruta}")
            continue

        mascara = _mascara_de_fila(fila, imagen)

        hashes.append(dhash(imagen))
        lista_features.append(extractor(imagen, mascara))

        if con_cnn:
            lado = max(CNN_IMAGE_SIZE)
            lista_imgs.append(a_cuadrado(imagen, lado, cv2.INTER_AREA, GRIS_RELLENO))
            # La mascara se rellena con 0: lo anadido es fondo, no fruta.
            lista_masks.append(a_cuadrado(mascara, lado, cv2.INTER_NEAREST, 0))

        filas_validas.append(i)

        if (len(filas_validas)) % 250 == 0:
            print(f"  ... {len(filas_validas)}/{len(df)} imagenes procesadas")

    df = df.loc[filas_validas].reset_index(drop=True)

    print("\nAgrupando imagenes casi duplicadas (dHash + nombre de archivo)...")
    claves = [clave_original(r) for r in df["filepath"].astype(str)]
    grupos = agrupar_duplicados(hashes, claves_extra=claves)
    n_grupos = int(len(set(grupos.tolist())))
    print(f"  grupos unicos: {n_grupos} para {len(df)} imagenes")
    print(f"  tasa de duplicacion aparente: {1 - n_grupos / max(len(df), 1):.1%}")

    df["dhash"] = [f"{h:016x}" for h in hashes]
    df["grupo_duplicado"] = grupos
    df["split"] = particionar_por_grupos(df, grupos, semilla=semilla)

    X = np.asarray(lista_features, dtype=np.float32)
    y_quality = df["quality"].map(QUALITY_TO_IDX).to_numpy(dtype=np.int64)
    y_size = df["size_label"].map(SIZE_TO_IDX).to_numpy(dtype=np.int64)
    y_fruit = df["fruit"].map(FRUIT_TO_IDX).to_numpy(dtype=np.int64)

    sufijo = "" if tipo_features == "enriched" else "_baseline"
    np.save(FEATURES_DIR / f"X{sufijo}.npy", X)
    np.save(FEATURES_DIR / "y_quality.npy", y_quality)
    np.save(FEATURES_DIR / "y_size.npy", y_size)
    np.save(FEATURES_DIR / "y_fruit.npy", y_fruit)
    np.save(FEATURES_DIR / "split.npy", df["split"].to_numpy().astype(str))
    np.save(FEATURES_DIR / "grupo.npy", grupos)

    if con_cnn:
        np.save(FEATURES_DIR / "X_cnn_bgr_uint8.npy", np.asarray(lista_imgs, dtype=np.uint8))
        np.save(FEATURES_DIR / "X_cnn_mask_uint8.npy", np.asarray(lista_masks, dtype=np.uint8))

    df.to_csv(FEATURES_DIR / "index.csv", index=False, encoding="utf-8")

    resumen = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "tipo_features": tipo_features,
        "n_imagenes": int(len(df)),
        "n_features": int(X.shape[1]),
        "nombres_features": nombres,
        "n_grupos_duplicados": n_grupos,
        "umbral_hamming": UMBRAL_HAMMING,
        "max_kaggle_por_clase": max_kaggle,
        "semilla": semilla,
        "cnn_image_size": list(CNN_IMAGE_SIZE),
        "conteo_split": df["split"].value_counts().to_dict(),
        "conteo_quality": df["quality"].value_counts().to_dict(),
        "conteo_size": df["size_label"].value_counts().to_dict(),
        "quality_por_split": (
            pd.crosstab(df["split"], df["quality"]).to_dict(orient="index")
        ),
    }
    with open(FEATURES_DIR / f"meta{sufijo}.json", "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)

    print("\n=== RESUMEN ===")
    print(f"X: {X.shape}  y_quality: {y_quality.shape}")
    print(df["split"].value_counts())
    print(pd.crosstab(df["split"], df["quality"]))
    print(f"\nArtefactos en: {FEATURES_DIR}")
    return resumen


def main() -> None:
    from src.utils.registro import registrar_ejecucion

    with registrar_ejecucion("build_features"):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--features",
            choices=("enriched", "baseline"),
            default="enriched",
            help="Extractor a usar: 'enriched' (93 dim, este proyecto) o 'baseline' (1028 dim)",
        )
        parser.add_argument("--no-cnn", action="store_true", help="No guardar los tensores de la CNN")
        parser.add_argument("--limit", type=int, default=None, help="Procesar solo N imagenes (pruebas)")
        parser.add_argument(
            "--max-kaggle",
            type=int,
            default=0,
            help="Maximo de imagenes de Kaggle por fruta x calidad (0 = sin limite)",
        )
        args = parser.parse_args()

        construir(
            tipo_features=args.features,
            con_cnn=not args.no_cnn,
            limite=args.limit,
            max_kaggle=args.max_kaggle,
        )


if __name__ == "__main__":
    main()
