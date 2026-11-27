# Reparto de trabajo del equipo

> Estado del proyecto y asignación de lo que falta. Cada tarea trae qué hay
> que entregar, cómo se sabe que está terminada y qué comando ejecutar.
>
> Última actualización del estado: ver `experiments/results/informe_consolidado.md`.

---

## 1. Qué hay hecho y quién lo hizo

| Bloque | Estado | Archivos |
|---|---|---|
| Estructura del repositorio y README inicial | Hecho | `README.md`, `LICENSE` |
| EDA y notebooks exploratorios | Hecho | `notebooks/01_analisis_exploratorio.ipynb`, `notebooks/01_eda_preprocesamiento.ipynb` |
| Consolidación del dataset y segmentación heurística | Hecho | `src/data/build_dataset.py` |
| Preprocesamiento, splits y features iniciales | Hecho | `src/data/preprocess.py` |
| Segmentación con rembg | Hecho (sin ejecutar: falta instalar `rembg`) | `src/data/segmentation_rembg.py` |
| Ingesta del material propio (HEIC, EXIF, filtros) | Hecho | `src/data/extract_kaggle.py`, `src/data/ingest_recolectado.py` |
| Características enriquecidas de 93 dimensiones | Hecho | `src/features/extractors.py` |
| Partición sin fuga por casi duplicados | Hecho | `src/features/build_features.py` |
| Modelos clásicos + ajuste de hiperparámetros | Hecho | `src/models/classic.py`, `src/training/train_classic.py` |
| CNN propia multitarea | Hecho | `src/models/cnn.py`, `src/training/train_cnn.py` |
| Métricas, figuras y evaluación consolidada | Hecho | `src/evaluation/` |
| Metodología, formulación matemática y ética | Hecho | `docs/01`, `docs/02`, `docs/03`, `docs/figuras/` |
| Despliegue (GUI) | Hecho, en EC2: ~1,4 s por imagen en CPU (t3.small), casi todo es rembg | `src/deployment/predict.py`, `src/deployment/app.py` |
| **Anotación de la clase `regular`** | **Pendiente** | — |
| **Informe final y video** | **Pendiente** | — |

---

## 2. Tareas pendientes, por persona

### Persona 1 — Recolección y anotación de datos

Es la tarea que más mueve los resultados: la clase `regular` tiene recall
0,585 frente a 0,95 de las otras dos, y es el cuello de botella del sistema.

**1.1 Tomar fotografías nuevas de fruta `regular`**

El conteo de archivos engaña. Lo que importa son las fotos originales: 148 de
las 854 son variantes generadas con aumentación (`_aug_NNN`), y en algunos
casos casi toda la carpeta sale de unas pocas fotos.

| Fruta | Archivos `regular` | **Fotos originales** | Falta tomar |
|---|---|---|---|
| limón/lima | 39 | **5** | **70** |
| guayaba | 62 | 62 | **50** |
| banano | 66 | 66 | **40** |
| manzana | 76 | 76 | hecho |
| naranja | 93 | 93 | hecho |
| **Total** | **360** | **337** | **160** |

> **Actualización.** La segunda recolección aportó 70 manzanas y 57 naranjas
> `regular`. Tras la revisión de etiquetas (16 reetiquetadas a `good` y 2
> excluidas por no ser frutas individuales), la clase `regular` pasó de 238 a
> 360 imágenes. Quedan pendientes limón, guayaba y banano.

Condiciones de captura (ver `docs/01_metodologia_crisp_dm.md` §3.3):

- Una fruta **distinta** por foto. Cinco ángulos de la misma fruta cuentan
  como uno solo: el agrupador de duplicados los une.
- La fruta debe ocupar algo más de la mitad del encuadre. Las fotos actuales
  tienen la fruta al 32 % del encuadre frente al 55 % de Kaggle, y por eso 602
  de 854 quedaron etiquetadas como "pequeño" sin serlo.
- Misma orientación en todas (vertical) y **misma distancia**, con marca en la
  mesa o un objeto de referencia (moneda de $500 = 23,7 mm).
- Fondo liso y contrastante. Nada de pasto ni mesones con textura.
- Sin zoom digital.
- **No aumentar las fotos a mano.** La aumentación la hace el código, solo
  sobre el conjunto de entrenamiento. Meter variantes en `data_raw/` infla los
  conteos y arriesga fuga entre particiones.

*Terminado cuando:* hay ~280 fotos nuevas, el script de ingesta las procesa
sin descartes y `regular` llega a ~470 fotos originales.

**1.2 Fotografías de `bad` para las frutas con poco soporte**

`banana_bad` quedó en 14 fotos originales y `lime_bad` en 7. Unas 25 por cada
una, con las mismas condiciones.

**1.3 Anotar la carpeta Mixed Quality de Kaggle**

El enunciado lo pide explícitamente. Ya hay 365 imágenes extraídas en
`data_raw/kaggle/archive/Mixed Qualit_Fruits/`. Advertencias: son de 192×256
px, traen 4-6 frutas por foto sobre pasto y hay ráfagas casi idénticas
(`IMG20200728125930`, `...125932`, `...125942` son la misma escena).

Procedimiento: recortar cada fruta individual, descartar las ráfagas
repetidas, etiquetar con el protocolo de `docs/01` §3.2 y guardar en
`data_raw/recolectado/<Calidad>_Quality_Fruits/<fruta>_<calidad>/`.

**1.4 Medir el acuerdo entre anotadores** — material ya preparado

Dos personas etiquetan las mismas 100 imágenes por separado y se calcula el
$\kappa$ de Cohen. Si sale por debajo de 0,60, el protocolo de anotación no es
suficientemente claro y hay que ajustarlo **antes** de seguir anotando. Sin
esta medición no se puede afirmar en el informe que las etiquetas son
confiables.

El ejercicio ya está armado con una muestra estratificada de 100 fotografías
propias, barajadas y sin mostrar su etiqueta actual:

- Hoja de contacto numerada: `docs/anotacion/acuerdo_hoja_de_contacto.jpg`
- Plantilla a llenar: `docs/anotacion/acuerdo_plantilla.csv`

Cada persona copia la plantilla con su nombre, la llena mirando **solo** la
hoja de contacto y sin consultar con la otra. Después:

```powershell
python -m src.evaluation.acuerdo calcular docs/anotacion/acuerdo_ana.csv docs/anotacion/acuerdo_juan.csv
```

Reporta el $\kappa$ simple, el $\kappa$ ponderado cuadrático (que penaliza más
confundir `good` con `bad` que `good` con `regular`, porque las clases están
ordenadas) y la matriz de acuerdo, que señala exactamente qué pares de clases
hay que precisar en el protocolo.

---

### Persona 2 — Despliegue (fase 6 de CRISP-DM)

Es la fase que falta por completo y es requisito del enunciado.

**2.1 Módulo de inferencia — `src/deployment/predict.py`**

```python
def predecir(imagen_o_ruta) -> dict:
    """Devuelve {calidad, confianza_calidad, tamano, confianza_tamano, mascara}."""
```

Requisito crítico: **debe reutilizar el mismo preprocesamiento del
entrenamiento**, importándolo de `src/features/` y `src/models/cnn.py`, no
reimplementándolo. Si la GUI preprocesa distinto a como se entrenó, el modelo
se degrada sin que nada avise (*training–serving skew*).

Piezas que ya existen y hay que llamar:
- `src.features.build_features.a_cuadrado` para el relleno a cuadrado.
- `src.features.extractors.extract_features` para las 93 características.
- `src.models.cnn.normalizar_lote` para la entrada de la red.
- Modelos: `experiments/checkpoints/mejor_classic_quality.joblib` y
  `experiments/checkpoints/cnn_mejor.pt`.

*Terminado cuando:* `predecir()` devuelve el mismo resultado que el pipeline de
entrenamiento sobre una imagen del conjunto de prueba.

**2.2 Interfaz — `src/deployment/app.py` (Streamlit)**

- Carga de archivo y captura con cámara (`st.camera_input`).
- Muestra la clase, **la probabilidad** y la máscara segmentada, para que el
  usuario vea qué miró el modelo.
- Aviso explícito cuando la confianza es baja, en vez de dar una respuesta
  falsamente segura.
- Botón para que el operario anule la predicción (decisión ética documentada
  en `docs/03`, dilema 7).

```powershell
streamlit run src/deployment/app.py
```

**2.3 Medir el tiempo de inferencia**

El criterio de éxito operativo es menos de 1 segundo por imagen en CPU.
Reportarlo en el informe.

---

### Persona 3 — Informe final y presentación

**3.1 Informe de máximo 7 páginas, en LaTeX, con las 9 secciones del enunciado**

Casi todo el contenido técnico ya está escrito y solo hay que condensarlo:

| Sección del informe | De dónde sale |
|---|---|
| Fundamentos teóricos | `docs/02_formulacion_matematica.md` §3, §4, §6 |
| Metodología | `docs/01_metodologia_crisp_dm.md` + `docs/figuras/crisp_dm_flujo.svg` |
| Resultados | `experiments/results/informe_consolidado.md` y las figuras SVG |
| Análisis de resultados | §4 de este documento (hallazgos) |
| Ética e impactos | `docs/03_etica_e_impactos.md` |

**3.2 Completar las citas pendientes**

Están marcadas como `[CITA PENDIENTE]` en los documentos:
- Pérdidas poscosecha de frutas en Colombia (FAO, DANE o DNP).
- Norma técnica de clasificación de calidad de frutas aplicable.
- Licencia exacta del dataset de Kaggle.
- Dos o tres trabajos publicados sobre este mismo dataset, para la comparación
  con la literatura que pide la rúbrica.

Formato IEEE, y solo material efectivamente leído.

**3.3 Video de máximo 10 minutos**

Contexto del problema, metodología, resultados y hallazgos principales.

---

### Persona 4 — Lo que ya está hecho (este bloque)

Modelado, evaluación y documentación metodológica. Queda por hacer:

- Ejecutar el experimento de 2 clases con las ~10.000 imágenes de Kaggle para
  la comparación con la literatura.
- Reentrenar cuando lleguen las fotos nuevas de `regular`.
- Pruebas automáticas en `tests/`.

---

## 3. Orden recomendado

```
Persona 1 toma las fotos  ─┐
                           ├─► reentrenar ──► informe (Persona 3)
Persona 2 arma la GUI   ───┘
```

Las fotos nuevas son lo que bloquea el resultado final: sin ellas, `regular`
se queda en recall 0,585 y ese número va al informe. La GUI puede construirse
en paralelo con los modelos actuales y cambiar el checkpoint al final.

---

## 4. Hallazgos que deben aparecer en el informe

Son los cinco puntos que diferencian este trabajo de uno que solo reporta
exactitud. Están medidos, no supuestos.

1. **37 % de las imágenes tiene casi duplicados.** Con partición aleatoria,
   copias de la misma fruta caen en entrenamiento y prueba a la vez y las
   métricas quedan infladas. Se resolvió agrupando por huella perceptual dHash
   y partiendo por grupos.
2. **La tarea de tamaño es circular.** La etiqueta se obtiene discretizando el
   diámetro relativo, que además es una de las 93 características de entrada.
   Por eso Random Forest alcanza $F_1 = 1{,}000$: reproduce una fórmula. La
   CNN, que no ve esa característica, baja a 0,688, lo que confirma el
   diagnóstico. Además el diámetro relativo depende de la distancia de la
   cámara, así que no mide tamaño físico.
3. **La clase intermedia es el cuello de botella.** Recall 0,585 frente a 0,95
   de `good` y `bad`. Reportar solo la exactitud (0,93) lo ocultaría.
4. **Los errores no cuestan lo mismo.** Se reporta el riesgo empírico con una
   matriz de costos asimétricos y el número de falsos `good` sobre `bad`.
5. **Hay cambio de dominio entre las dos fuentes.** Las imágenes de Kaggle son
   de estudio y las propias de plaza de mercado. Se mide el desempeño por
   fuente, porque un promedio global lo escondería.
6. **Se detectó y corrigió un confusor por condición de captura.** Toda la
   segunda recolección se hizo en un supermercado y, al principio, toda ella
   pertenecía a la clase `regular`. Un clasificador separaba esas imágenes de
   las `good` con 81,7 % de exactitud usando forma e iluminación, no defectos:
   la fracción de píxeles oscuros incluso iba en sentido contrario al
   esperado. Se corrigió reetiquetando, de modo que la clase `good` también
   contiene fotografías de esa sesión. Detalle en `docs/01` §2.4.

---

## 5. Comandos de referencia

```powershell
pip install -r requirements.txt
```

```powershell
python -m src.data.ingest_recolectado --zip "C:/ruta/fotos.zip"
```

```powershell
python -m src.data.build_dataset; python -m src.features.build_features
```

```powershell
python -m src.training.train_classic --task both; python -m src.training.train_cnn; python -m src.evaluation.evaluate
```
