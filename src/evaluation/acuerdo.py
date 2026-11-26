"""
Acuerdo entre anotadores: preparacion del ejercicio y calculo del kappa.

Para que
--------
Las etiquetas de calidad no son una medicion fisica sino un acuerdo humano.
Afirmar en el informe que son confiables exige demostrarlo con un numero, y el
estandar para eso es el coeficiente kappa de Cohen:

    kappa = (p_o - p_e) / (1 - p_e)

donde p_o es la proporcion de imagenes en que los dos anotadores coinciden y
p_e la proporcion que coincidiria por puro azar, dadas las frecuencias con que
cada uno usa cada clase:

    p_e = sum_k  P(anotador A elige k) * P(anotador B elige k)

Un kappa de 1 es acuerdo perfecto, 0 es el que se obtendria lanzando una
moneda con esas mismas frecuencias, y negativo es peor que el azar. La escala
de Landis y Koch, que es la referencia habitual:

    < 0,00  sin acuerdo        0,41-0,60  moderado
    0,00-0,20  leve            0,61-0,80  sustancial
    0,21-0,40  aceptable       0,81-1,00  casi perfecto

Si el kappa del grupo sale por debajo de 0,60, el protocolo de anotacion de
`docs/01` no es lo bastante operativo y hay que precisarlo ANTES de seguir
etiquetando: de nada sirve un modelo entrenado sobre etiquetas en las que ni
siquiera las personas se ponen de acuerdo.

Como kappa penaliza el desacuerdo sin mirar cuan lejos esta, y las clases de
calidad estan ordenadas (good < regular < bad), se reporta tambien el **kappa
ponderado cuadratico**, que castiga mas confundir `good` con `bad` que `good`
con `regular`.

Uso
---
    # 1. Preparar el ejercicio (genera hojas y plantillas)
    python -m src.evaluation.acuerdo preparar --n 100

    # 2. Cada persona llena su CSV, por separado y sin mirar el del otro

    # 3. Calcular el acuerdo
    python -m src.evaluation.acuerdo calcular ^
        docs/anotacion/acuerdo_ana.csv docs/anotacion/acuerdo_juan.csv

Referencia: J. R. Landis y G. G. Koch, "The measurement of observer agreement
for categorical data", Biometrics, vol. 33, no. 1, pp. 159-174, 1977.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import DOCS, LABELS_CSV, PROJECT_ROOT, QUALITY_CLASSES, RANDOM_STATE

CARPETA = DOCS / "anotacion"
HOJA = CARPETA / "acuerdo_hoja_de_contacto"
PLANTILLA = CARPETA / "acuerdo_plantilla.csv"
LISTADO = CARPETA / "acuerdo_imagenes.csv"


# ============================================================
# 1. PREPARAR EL EJERCICIO
# ============================================================

def preparar(n: int = 100, semilla: int = RANDOM_STATE, columnas: int = 10) -> Path:
    """
    Elige n imagenes de la recoleccion propia, genera una hoja de contacto
    numerada y una plantilla CSV en blanco para cada anotador.

    Se muestrean solo las fotografias propias porque son las que el grupo
    etiqueto: medir el acuerdo sobre las de Kaggle mediria el criterio de
    quien publico ese dataset, no el del equipo. El muestreo es estratificado
    por calidad para que la clase intermedia, que es la dificil, este
    representada.

    La hoja NO muestra la etiqueta actual. Si la mostrara, el ejercicio
    dejaria de medir acuerdo y mediria obediencia.
    """
    import cv2

    if not LABELS_CSV.exists():
        raise FileNotFoundError(
            f"No existe {LABELS_CSV}. Ejecuta antes: python -m src.data.build_dataset"
        )

    df = pd.read_csv(LABELS_CSV)
    propias = df[df["source"] == "recolectado"] if "source" in df.columns else df
    if propias.empty:
        raise ValueError("No hay fotografias propias en labels.csv")

    por_clase = max(1, n // len(QUALITY_CLASSES))
    partes = []
    for clase in QUALITY_CLASSES:
        sub = propias[propias["quality"] == clase]
        if sub.empty:
            continue
        partes.append(sub.sample(n=min(por_clase, len(sub)), random_state=semilla))

    muestra = pd.concat(partes)
    if len(muestra) < n:  # completar si alguna clase tenia poco soporte
        resto = propias.drop(muestra.index)
        faltan = min(n - len(muestra), len(resto))
        if faltan > 0:
            muestra = pd.concat([muestra, resto.sample(n=faltan, random_state=semilla)])

    # Se barajan para que el orden no delate la clase.
    muestra = muestra.sample(frac=1.0, random_state=semilla).reset_index(drop=True)
    CARPETA.mkdir(parents=True, exist_ok=True)

    celdas = []
    registros = []
    for i, fila in muestra.iterrows():
        numero = i + 1
        imagen = cv2.imread(str(PROJECT_ROOT / fila["filepath"]))
        if imagen is None:
            continue

        celda = cv2.resize(imagen, (220, 220))
        cv2.rectangle(celda, (0, 0), (52, 26), (255, 255, 255), -1)
        cv2.putText(celda, str(numero), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.rectangle(celda, (0, 0), (219, 219), (60, 60, 60), 1)
        celdas.append(celda)
        registros.append(
            {"numero": numero, "filepath": fila["filepath"], "fruta": fila["fruit"]}
        )

    filas_img = []
    for i in range(0, len(celdas), columnas):
        bloque = celdas[i : i + columnas]
        while len(bloque) < columnas:
            bloque.append(np.full((220, 220, 3), 245, dtype=np.uint8))
        filas_img.append(np.hstack(bloque))

    rejilla = np.vstack(filas_img)
    banda = np.full((44, rejilla.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        banda,
        "ACUERDO ENTRE ANOTADORES - etiqueta cada numero como good, regular o bad",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    hoja = HOJA.with_suffix(".jpg")
    cv2.imwrite(str(hoja), np.vstack([banda, rejilla]))

    with open(LISTADO, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=["numero", "filepath", "fruta"])
        escritor.writeheader()
        escritor.writerows(registros)

    with open(PLANTILLA, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(["numero", "etiqueta"])
        for r in registros:
            escritor.writerow([r["numero"], ""])

    print(f"Imagenes seleccionadas: {len(registros)}")
    print(f"Hoja de contacto:  {hoja}")
    print(f"Plantilla a llenar: {PLANTILLA}")
    print(f"Listado (no mirar antes de anotar): {LISTADO}")
    print(
        "\nCada persona copia la plantilla con su nombre, por ejemplo\n"
        "  docs/anotacion/acuerdo_ana.csv  y  docs/anotacion/acuerdo_juan.csv\n"
        "la llena mirando SOLO la hoja de contacto, y sin consultar con la otra."
    )
    return hoja


# ============================================================
# 2. CALCULAR EL ACUERDO
# ============================================================

def _matriz_pesos(k: int) -> np.ndarray:
    """Pesos cuadraticos: w_ij = 1 - ((i-j)/(k-1))^2."""
    i, j = np.indices((k, k))
    return 1.0 - ((i - j) / (k - 1)) ** 2


def leer_anotacion(ruta: Path) -> pd.Series:
    """
    Lee el CSV de un anotador tolerando las variantes que produce Excel.

    Segun la configuracion regional, Excel guarda los CSV separados por coma o
    por punto y coma, y a veces mezcla ambos en la misma linea. Tambien es
    frecuente que las etiquetas queden en mayusculas o con espacios. En lugar
    de exigir un formato exacto a quien anota, se normaliza aqui: se parte
    cada linea por cualquiera de los separadores habituales, se descarta la
    cabecera y se pasa la etiqueta a minusculas.

    Devuelve una serie indexada por numero de imagen.
    """
    import re

    numeros, etiquetas = [], []
    for linea in Path(ruta).read_text(encoding="utf-8-sig").splitlines():
        campos = [c.strip() for c in re.split(r"[;,\t|]", linea.strip()) if c.strip()]
        if len(campos) < 2 or not campos[0].isdigit():
            continue  # cabecera, linea vacia o fila sin etiqueta
        numeros.append(int(campos[0]))
        etiquetas.append(campos[1].lower())

    if not numeros:
        raise ValueError(f"No se pudo leer ninguna anotacion de {ruta}")

    return pd.Series(etiquetas, index=numeros, name=Path(ruta).stem)


def etiquetas_del_dataset() -> pd.Series:
    """
    Etiquetas que el dataset tiene hoy para las imagenes del ejercicio.

    Permite comparar a un anotador contra las etiquetas vigentes. No sustituye
    al acuerdo entre dos personas (el dataset no es un anotador independiente:
    sus etiquetas vienen del propio equipo), pero sirve para detectar cuanto
    ruido de etiqueta arrastra el material.
    """
    if not LISTADO.exists():
        raise FileNotFoundError(f"No existe {LISTADO}. Ejecuta antes: acuerdo preparar")

    listado = pd.read_csv(LISTADO)
    df = pd.read_csv(LABELS_CSV).set_index("filepath")["quality"]
    filas = {
        int(r["numero"]): str(df.get(r["filepath"], "")).lower()
        for _, r in listado.iterrows()
    }
    return pd.Series(filas, name="dataset")


def calcular(ruta_a: Path, ruta_b: Path) -> dict:
    """
    Calcula el kappa simple y el ponderado cuadratico entre dos anotadores.

    Cualquiera de las dos rutas puede ser la palabra 'dataset', en cuyo caso
    se usan las etiquetas vigentes del proyecto en lugar de un CSV.
    """
    a = etiquetas_del_dataset() if str(ruta_a) == "dataset" else leer_anotacion(ruta_a)
    b = etiquetas_del_dataset() if str(ruta_b) == "dataset" else leer_anotacion(ruta_b)
    nombre_a, nombre_b = a.name, b.name

    comunes = sorted(set(a.index) & set(b.index))
    a, b = a.loc[comunes], b.loc[comunes]
    validas = a.isin(QUALITY_CLASSES) & b.isin(QUALITY_CLASSES)
    descartadas = int((~validas).sum())
    a, b = a[validas], b[validas]

    if len(a) == 0:
        raise ValueError("No hay anotaciones validas en comun")

    idx = {c: i for i, c in enumerate(QUALITY_CLASSES)}
    ia = a.map(idx).to_numpy()
    ib = b.map(idx).to_numpy()
    k = len(QUALITY_CLASSES)

    observada = np.zeros((k, k))
    for x, y in zip(ia, ib):
        observada[x, y] += 1
    observada /= observada.sum()

    marginal_a = observada.sum(axis=1)
    marginal_b = observada.sum(axis=0)
    esperada = np.outer(marginal_a, marginal_b)

    p_o = float(np.trace(observada))
    p_e = float(np.trace(esperada))
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0

    w = _matriz_pesos(k)
    po_w = float((w * observada).sum())
    pe_w = float((w * esperada).sum())
    kappa_w = (po_w - pe_w) / (1 - pe_w) if pe_w < 1 else 1.0

    def interpretar(valor: float) -> str:
        for limite, texto in ((0.0, "sin acuerdo"), (0.21, "leve"), (0.41, "aceptable"),
                              (0.61, "moderado"), (0.81, "sustancial"), (1.01, "casi perfecto")):
            if valor < limite:
                return texto
        return "casi perfecto"

    resultado = {
        "n_comparadas": int(len(a)),
        "n_descartadas": descartadas,
        "acuerdo_observado": p_o,
        "acuerdo_esperado_por_azar": p_e,
        "kappa": kappa,
        "kappa_ponderado": kappa_w,
        "interpretacion": interpretar(kappa),
        "matriz": (observada * len(a)).astype(int).tolist(),
        "clases": list(QUALITY_CLASSES),
        "anotador_a": str(nombre_a),
        "anotador_b": str(nombre_b),
    }

    print(f"Imagenes comparadas: {resultado['n_comparadas']}"
          f"  (descartadas por etiqueta invalida: {descartadas})")
    print(f"Acuerdo observado:   {p_o:.3f}")
    print(f"Acuerdo por azar:    {p_e:.3f}")
    print(f"Kappa de Cohen:      {kappa:.3f}  ({resultado['interpretacion']})")
    print(f"Kappa ponderado:     {kappa_w:.3f}")
    print("\nMatriz de acuerdo (filas = anotador A, columnas = anotador B):")
    print(pd.DataFrame(resultado["matriz"], index=QUALITY_CLASSES, columns=QUALITY_CLASSES))

    if kappa < 0.60:
        print(
            "\n[ATENCION] Por debajo de 0,60 el acuerdo no es suficiente. Antes de\n"
            "seguir etiquetando hay que precisar el protocolo de docs/01 seccion 3.2,\n"
            "revisando sobre todo las celdas fuera de la diagonal."
        )
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="accion", required=True)

    p_prep = sub.add_parser("preparar", help="Genera la hoja de contacto y las plantillas")
    p_prep.add_argument("--n", type=int, default=100)

    p_calc = sub.add_parser("calcular", help="Calcula el kappa entre dos anotadores")
    p_calc.add_argument("anotador_a", type=Path, help="CSV del anotador, o la palabra dataset")
    p_calc.add_argument("anotador_b", type=Path, help="CSV del anotador, o la palabra dataset")

    args = parser.parse_args()
    if args.accion == "preparar":
        preparar(n=args.n)
    else:
        calcular(args.anotador_a, args.anotador_b)


if __name__ == "__main__":
    main()
