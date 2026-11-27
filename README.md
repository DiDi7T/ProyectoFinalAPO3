# Clasificación automática de calidad de frutas por visión por computadora

Proyecto final de **Algoritmos y Programación III** — Universidad Icesi, Cali,
Colombia. Semestre 2026-2.

En mercados y agroindustrias la clasificación manual de productos frescos
según su madurez, tamaño o defectos visibles es lenta, subjetiva y propensa a
errores, lo que genera pérdidas económicas, desperdicio de alimentos y falta
de estandarización. Este proyecto construye un sistema que, a partir de una
sola fotografía de una fruta, predice su **categoría de calidad** y su
**tamaño relativo**.

#### Estado del proyecto: activo

## Miembros

Proyecto desarrollado por cuatro estudiantes de Ingeniería de Sistemas.

| Nombre | Autor en el historial | Bloque de trabajo |
|---|---|---|
| Anderson Romero | `Anderson` | EDA y preprocesamiento |
| Diana Garzón Toro | `DiDi7T` | Segmentación y análisis exploratorio |
| Juan José Reyes | `Juan José` | Modelado, evaluación y metodología |
| Juan Rubio Arboleda | `Juan Rubio` | Despliegue e interfaz |

## Objetivo

Desarrollar un sistema automático de clasificación de calidad basado en visión
por computadora, capaz de analizar imágenes de frutas, asignar una categoría
de calidad (`good`, `regular`, `bad`) y estimar el tamaño relativo del
producto, siguiendo la metodología CRISP-DM.

## Métodos

- Segmentación por saturación con umbral de Otsu y morfología matemática,
  y segmentación neural con rembg/U²-Net para las fotografías propias
- Ingeniería de características interpretables (color, forma, textura LBP,
  indicadores de defecto)
- Machine learning clásico: regresión logística, KNN, SVM-RBF, Random Forest,
  gradient boosting, con ajuste de hiperparámetros por validación cruzada
- Red neuronal convolucional propia, multitarea, entrenada desde cero
- Detección de casi duplicados con huella perceptual y partición por grupos

## Tecnologías

Python 3.13 · NumPy · pandas · OpenCV · scikit-learn · PyTorch · Pillow ·
pillow-heif · Matplotlib · Streamlit · pytest

## Datos

| Fuente | Imágenes | Clases de calidad |
|---|---|---|
| [Fruit Quality Classification (Kaggle)](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification) | 19.526 disponibles, muestra estratificada en uso | `good`, `bad` |
| Recolección propia del grupo | 1.001 | `good`, `regular`, `bad` |

Las imágenes **no se versionan en git** (ver `.gitignore`). Se reconstruyen
desde los archivos originales con los comandos de la sección siguiente, de
forma reproducible gracias a la semilla fija.

Cinco frutas en alcance: manzana, banano, guayaba, lima/limón y naranja.

## Cómo reproducir

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Preparar los datos (los dos primeros comandos solo la primera vez):

```powershell
python -m src.data.extract_kaggle --zip "C:/ruta/kaggle.zip" --per-class 1000
```

```powershell
python -m src.data.ingest_recolectado --zip "C:/ruta/fotos_propias.zip"
```

```powershell
python -m src.data.build_dataset
```

```powershell
python -m src.features.build_features
```

Entrenar y evaluar:

```powershell
python -m src.training.train_classic --task both
```

```powershell
python -m src.training.train_cnn --epocas 30
```

```powershell
python -m src.evaluation.evaluate
```

Abrir la interfaz gráfica (necesita los modelos en `experiments/checkpoints/`):

```powershell
streamlit run src/deployment/app.py
```

O predecir una imagen desde consola:

```powershell
python -m src.deployment.predict ruta/a/foto.jpg --modelo cnn
```

Los resultados quedan en `experiments/results/` (métricas en JSON y CSV,
figuras en SVG y PNG) y los modelos entrenados en `experiments/checkpoints/`.

## Estructura del repositorio

```
src/
  data/        extracción, ingesta, consolidación y segmentación
  features/    extracción de características y construcción de particiones
  models/      catálogo de modelos clásicos y arquitectura de la CNN
  training/    entrenamiento y ajuste de hiperparámetros
  evaluation/  métricas, figuras e informe consolidado
  deployment/  inferencia (predict.py) e interfaz gráfica (app.py)
docs/          metodología, formulación matemática y ética
notebooks/     análisis exploratorio y preprocesamiento
experiments/   modelos entrenados, logs y resultados
tests/         pruebas automáticas
```

## Documentación

| Documento | Contenido |
|---|---|
| [Metodología CRISP-DM](docs/01_metodologia_crisp_dm.md) | Las 6 fases instanciadas, decisiones y protocolo de anotación |
| [Formulación matemática](docs/02_formulacion_matematica.md) | Planteamiento formal, ecuaciones de cada bloque y métricas |
| [Ética e impactos](docs/03_etica_e_impactos.md) | Dilemas éticos y análisis de impactos |
| [Diagrama de flujo](docs/figuras/crisp_dm_flujo.svg) | CRISP-DM adaptado, formato vectorial |
| [Resultados consolidados](experiments/results/informe_consolidado.md) | Generado automáticamente por la evaluación |

## Decisiones metodológicas destacadas

- **Partición por grupos de casi duplicados.** El 49,8 % de las imágenes
  tiene al menos una copia casi idéntica. Con partición aleatoria, copias de
  la misma fruta caen en entrenamiento y prueba a la vez y las métricas
  quedan infladas.
- **F1 macro como métrica de selección**, no exactitud: la clase `regular` es
  minoritaria y es justamente la que importa detectar.
- **Costos asimétricos.** Clasificar fruta mala como buena cuesta más que lo
  contrario, y se reporta por separado.
- **Limitación declarada.** La estimación de tamaño depende de la distancia de
  la cámara y por tanto no mide tamaño físico. Documentado en `docs/02` §5.

## Licencia

Ver [LICENSE](LICENSE). El dataset de Kaggle conserva la licencia de su autor
original y se cita como fuente.
