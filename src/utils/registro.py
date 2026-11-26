"""
Registro de ejecuciones: deja constancia de cada entrenamiento.

Sin esto, la salida de consola de un entrenamiento se pierde al cerrar la
terminal y no queda rastro de con que argumentos se genero cada resultado.
Para el informe final hace falta poder responder preguntas como "con que
version de los datos se obtuvo este 0,857" meses despues.

Cada ejecucion deja dos cosas en experiments/logs/:

  <script>_<marca de tiempo>.log   la salida completa de consola
  historial.csv                     una fila por ejecucion, con la fecha, el
                                    comando, la duracion y el resultado

Los logs NO se versionan en git (ver .gitignore): son voluminosos, cambian en
cada corrida y no aportan al repositorio. Lo que si se versiona es
experiments/results/, que contiene las metricas y figuras definitivas.

Uso:

    from src.utils.registro import registrar_ejecucion

    with registrar_ejecucion("train_classic"):
        ...
"""

from __future__ import annotations

import csv
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from src.utils.config import LOGS

HISTORIAL = LOGS / "historial.csv"
CAMPOS = ["fecha", "script", "comando", "segundos", "estado", "log"]


class _Duplicador:
    """Escribe en la consola y en el archivo de log a la vez (tipo 'tee')."""

    def __init__(self, consola, archivo) -> None:
        self.consola = consola
        self.archivo = archivo

    def write(self, texto: str) -> int:
        self.consola.write(texto)
        self.archivo.write(texto)
        return len(texto)

    def flush(self) -> None:
        self.consola.flush()
        self.archivo.flush()

    def isatty(self) -> bool:
        return getattr(self.consola, "isatty", lambda: False)()


def _anotar_historial(fila: dict) -> None:
    existe = HISTORIAL.exists()
    with open(HISTORIAL, "a", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=CAMPOS)
        if not existe:
            escritor.writeheader()
        escritor.writerow(fila)


@contextmanager
def registrar_ejecucion(nombre: str):
    """
    Duplica la salida estandar a un archivo y anota la ejecucion en el
    historial. Si el bloque lanza una excepcion, queda registrada como
    'error' y la excepcion se vuelve a propagar.
    """
    LOGS.mkdir(parents=True, exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = LOGS / f"{nombre}_{marca}.log"

    comando = " ".join(sys.argv)
    inicio = time.perf_counter()
    estado = "ok"

    archivo = open(ruta, "w", encoding="utf-8")
    stdout_original, stderr_original = sys.stdout, sys.stderr
    sys.stdout = _Duplicador(stdout_original, archivo)
    sys.stderr = _Duplicador(stderr_original, archivo)

    print(f"# ejecucion: {comando}")
    print(f"# inicio:    {datetime.now().isoformat(timespec='seconds')}")

    try:
        yield ruta
    except BaseException as exc:
        estado = f"error: {type(exc).__name__}"
        raise
    finally:
        segundos = round(time.perf_counter() - inicio, 1)
        print(f"# fin: {estado} en {segundos} s")
        sys.stdout, sys.stderr = stdout_original, stderr_original
        archivo.close()

        _anotar_historial(
            {
                "fecha": datetime.now().isoformat(timespec="seconds"),
                "script": nombre,
                "comando": comando,
                "segundos": segundos,
                "estado": estado,
                "log": ruta.name,
            }
        )
        print(f"Log de la ejecucion: {ruta}")
