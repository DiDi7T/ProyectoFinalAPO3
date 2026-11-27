# Correcciones al borrador del informe

Revisión del borrador IEEE (7 páginas) contra los resultados vigentes del
repositorio, a la fecha del commit `0f7f445`. Cada cifra de aquí sale de
`experiments/results/` y es reproducible con los comandos de
`docs/04_reparto_equipo.md` §5.

**Causa de la mayoría de las diferencias:** el borrador se escribió con los
resultados de antes del commit `b891230`, cuando el modelo de calidad
seleccionado era SVM-RBF y el de tamaño Random Forest. El criterio de
selección cambió a la media de validación cruzada y ahora gana
HistGradientBoosting en las dos tareas.

---

## 1. Bloqueantes

### 1.1 El resumen es de otra versión del proyecto

El resumen habla de «sistema en desarrollo», «11 544 etiquetas: 11 325 de la
fuente pública y 219 recolectadas», «las imágenes de Kaggle no se incluyen en
el ZIP revisado» y dice que «la evaluación comparativa de los clasificadores,
el despliegue y la integración definitiva de las ramas permanecen pendientes
de documentar».

Nada de eso corresponde al cuerpo del informe, que ya reporta la comparación
completa y un despliegue funcionando. Es el resumen de una *revisión de un
ZIP intermedio*, no del proyecto. Hay que reescribirlo con las cifras
definitivas: 3.501 imágenes (2.500 de Kaggle y 1.001 propias), F1 macro de
0,9119 en calidad y 0,9476 en tamaño, segmentación con u2net, y despliegue
en Streamlit.

### 1.2 La Tabla I contiene los datos de tamaño, no de calidad

La Tabla I se titula «Desempeño de los modelos en clasificación de calidad»,
pero sus siete filas coinciden dígito por dígito con
`experiments/results/consolidado_size.csv`, incluida la columna de MAE
ordinal, que solo aplica a tamaño. **La tabla de calidad falta por
completo**, y la Tabla II («tamaño») es una copia truncada de la misma tabla.

Las tablas correctas están en §3 de este documento.

### 1.3 El modelo de tamaño ya no es Random Forest

El borrador afirma seis veces que Random Forest es el modelo de tamaño
(§III.E, §IV.B, §V.B, §V.D, §V.I y §VI). El modelo seleccionado y el
checkpoint versionado en `experiments/checkpoints/mejor_classic_size.joblib`
son **HistGradientBoosting**:

| Modelo | CV (5 pliegues) | Prueba |
|---|---|---|
| **HistGradientBoosting** | **0,9616 ± 0,0082** | 0,9476 |
| Random Forest | 0,9545 ± 0,0117 | 0,9520 |

Random Forest gana en prueba por 0,004, pero la prueba no selecciona. Si el
informe presenta Random Forest, no describe el sistema que se entrega.

**Cuidado con la figura `comparacion_size.png`.** Hasta el commit `7dc571b`
graficaba el F1 de la *partición de validación*, donde rf obtiene 0,9550 y
hgb 0,9541: la barra de rf salía más alta y contradecía al texto. Ya está
corregida y ahora grafica el criterio de selección, la media de los cinco
pliegues, con la desviación entre pliegues como barra de error y el modelo
elegido resaltado. **Si alguien tiene una copia anterior de esa imagen, hay
que reemplazarla.**

La versión corregida de `comparacion_quality.png` vale la pena en el
informe por sí misma: las barras de error de knn, svm_rbf, rf y hgb se
solapan visiblemente, que es exactamente el argumento de §1.4.

### 1.4 Confunde la validación cruzada con la partición de validación

§IV.A y §V.A dicen que HistGradientBoosting «obtuvo el mayor F1 macro en
validación, con 0,8886». 0,8886 es la **media de 5 pliegues de validación
cruzada**. En la partición de validación HGB obtiene 0,9148 y **gana
SVM-RBF con 0,9460**.

Más grave: §V.A justifica la elección «porque obtuvo el mejor F1 macro tanto
en validación como en prueba». Seleccionar con el conjunto de prueba es
precisamente lo que el propio informe prohíbe en §II.C, y aquí queda
confesado por escrito.

**Redacción correcta** (el argumento está en `docs/01` §4.2 y es más fuerte,
porque no toca el conjunto de prueba):

> La selección se hizo por la media de F1 macro en validación cruzada de
> cinco pliegues por grupos, y no por la partición de validación. La
> desviación entre pliegues del mejor modelo (0,0357) y la del segundo
> (0,0482) son un orden de magnitud mayores que la diferencia entre ambos
> (0,0063), de modo que una única partición de 720 imágenes decidiría por
> ruido. De hecho la partición de validación habría elegido SVM-RBF, el
> modelo con mayor varianza entre pliegues, que luego queda 2,6 puntos por
> debajo en prueba.

El JSON de resultados guarda las dos decisiones en los campos
`modelo_seleccionado` y `modelo_que_elegiria_la_particion_de_validacion`,
así que la afirmación es auditable.

---

## 2. Cifras desactualizadas

Todas son de la corrida anterior a `b891230`.

| Dato | Borrador | Vigente | Fuente |
|---|---|---|---|
| Calidad, exactitud en Kaggle | 0,9355 | **0,9501** | `por_fuente_quality.csv` |
| Calidad, exactitud en propias | 0,8443 | **0,8770** | `por_fuente_quality.csv` |
| Tamaño, exactitud en Kaggle | 0,9971 | **0,9912** | `por_fuente_size.csv` |
| Lima, F1 macro en calidad | 0,6590 | **0,6251** | `por_fruta_quality.csv` |
| Lima, errores en calidad | 13 | **8** | `por_fruta_quality.csv` |
| Lima, F1 macro en tamaño | 0,6558 | **0,6467** | `por_fruta_size.csv` |
| $\kappa$ ponderado cuadrático | 0,554 | **0,720** | `docs/01` §3.3.1 |
| Modelo de segmentación neural | BRIA RMBG 2.0 | **u2net** | `src/features/segmentacion.py:55` |

Sobre la última fila: el informe menciona «BRIA RMBG 2.0» tres veces
(resumen, §II.A y §III.C). El pipeline que produjo estos resultados usa
`rembg.new_session("u2net")`. Si alguien probó BRIA en un cuaderno aparte,
no es lo que generó las cifras reportadas.

Cifras del borrador que **sí** están correctas y no hay que tocar: los
hiperparámetros finales de los seis modelos (§III.D, verificados uno por
uno), las 3.501 imágenes y su composición 1.672/346/1.483, las particiones
2.318/720/463, los 304.742 parámetros de la CNN, la brecha de 0,0873, las
12 confusiones mala→buena, el F1 de 0,6280 de la ablación con máscara en
calidad y el de 0,7097 en tamaño, y guayaba con 1,0000 en tamaño.

---

## 3. Tablas corregidas

### Tabla I. Desempeño en clasificación de calidad (conjunto de prueba, n = 463)

| Modelo | F1 macro CV | F1 macro val. | Exactitud | Exact. balanceada | F1 macro | MCC | Falsos buenos |
|---|---|---|---|---|---|---|---|
| **HistGradientBoosting** | **0,8886** | 0,9148 | 0,9309 | 0,9020 | **0,9119** | 0,8805 | 12 |
| SVM-RBF | 0,8823 | 0,9460 | 0,9114 | 0,8875 | 0,8854 | 0,8477 | 13 |
| KNN | 0,8778 | 0,8879 | 0,9114 | 0,8943 | 0,8913 | 0,8484 | 11 |
| Random Forest | 0,8667 | 0,8850 | 0,9028 | 0,8636 | 0,8663 | 0,8325 | 16 |
| Regresión logística | 0,7864 | 0,8234 | 0,8251 | 0,8373 | 0,7844 | 0,7157 | 17 |
| CNN propia | — | 0,8914 | 0,8683 | 0,9037 | 0,8304 | 0,7922 | 13 |
| CNN, ablación sin fondo | — | — | 0,6695 | 0,7251 | 0,6280 | 0,5280 | 11 |
| DummyClassifier | 0,2139 | 0,2193 | 0,4838 | 0,3333 | 0,2174 | 0,0000 | 194 |

La CNN no tiene celda de validación cruzada porque no pasa por ella: su
ajuste se controla con la partición de validación y parada temprana. Ponerla
en esa columna la haría comparable con el criterio de selección de los
modelos clásicos, y no lo es.

### Tabla II. Desempeño en estimación de tamaño relativo (prueba, n = 463)

| Modelo | F1 macro CV | F1 macro val. | Exactitud | Exact. balanceada | F1 macro | MAE ordinal |
|---|---|---|---|---|---|---|
| **HistGradientBoosting** | **0,9616** | 0,9541 | 0,9460 | 0,9549 | 0,9476 | 0,0562 |
| Random Forest | 0,9545 | 0,9550 | 0,9503 | 0,9583 | 0,9520 | 0,0518 |
| Regresión logística | 0,8906 | 0,8760 | 0,8812 | 0,8961 | 0,8851 | 0,1210 |
| SVM-RBF | 0,8595 | 0,8135 | 0,8013 | 0,8143 | 0,8064 | 0,2009 |
| CNN propia | — | — | 0,7689 | 0,7781 | 0,7757 | 0,2333 |
| KNN | 0,7582 | 0,6608 | 0,7451 | 0,7496 | 0,7487 | 0,2592 |
| DummyClassifier | 0,1948 | 0,2147 | 0,4320 | 0,3333 | 0,2011 | 0,5680 |

### Tabla III. Desempeño de calidad por fuente

| Fuente | n | Exactitud | Exact. balanceada | F1 macro | Clases presentes |
|---|---|---|---|---|---|
| kaggle | 341 | 0,9501 | 0,9494 | 0,9499 | `good`, `bad` |
| recolectado | 122 | 0,8770 | 0,8803 | 0,8762 | `good`, `regular`, `bad` |
| **Todas** | 463 | 0,9309 | 0,9020 | 0,9119 | las tres |

La fila de Kaggle solo tiene dos clases: el dataset público no distingue
`regular`. La comparación entre fuentes no es a igualdad de dificultad y
conviene decirlo.

### Tabla IV. Desempeño de tamaño por fuente

| Fuente | n | Exactitud | Exact. balanceada | F1 macro |
|---|---|---|---|---|
| kaggle | 341 | 0,9912 | 0,9935 | 0,9912 |
| recolectado | 122 | 0,8197 | **0,5288** | **0,5349** |
| **Todas** | 463 | 0,9460 | 0,9549 | 0,9476 |

### Tabla V. Desempeño de calidad por fruta

| Fruta | n | Exactitud | F1 macro | Errores |
|---|---|---|---|---|
| banano | 88 | 0,9545 | 0,9654 | 4 |
| guayaba | 93 | 0,9462 | 0,9388 | 5 |
| manzana | 99 | 0,9192 | 0,9272 | 8 |
| naranja | 92 | 0,9239 | 0,8984 | 7 |
| lima | 91 | 0,9121 | **0,6251** | 8 |

### Tabla VI. Desempeño de tamaño por fruta

| Fruta | n | Exactitud | F1 macro | Errores |
|---|---|---|---|---|
| guayaba | 93 | 1,0000 | 1,0000 | 0 |
| naranja | 92 | 0,9348 | 0,9272 | 6 |
| banano | 88 | 0,9318 | 0,9255 | 6 |
| manzana | 99 | 0,9192 | 0,8955 | 8 |
| lima | 91 | 0,9451 | **0,6467** | 5 |

En lima el contraste entre exactitud y F1 macro es el patrón que conviene
explicar: acierta la clase mayoritaria y falla las minoritarias.

---

## 4. Omisiones que cuestan nota

### 4.1 El acuerdo entre anotadores está contado a la baja

El borrador reporta solo $\kappa = 0{,}470$ y un ponderado equivocado de
0,554, con lo que el informe se autoinflige la conclusión de que las
etiquetas no son confiables. Los tres números de `docs/01` §3.3.1 cuentan
otra historia, y es defendible:

| Qué se mide | $\kappa$ | Acuerdo |
|---|---|---|
| Las tres clases | 0,470 | 65 % |
| **Vendible (`good`+`regular`) vs descarte (`bad`)** | **0,828** | **93 %** |
| Primera calidad (`good`) vs el resto | 0,401 | 71 % |
| Las tres clases, ponderado cuadrático | 0,720 | — |

La decisión que de verdad importa para el negocio —si la fruta se vende o se
descarta— tiene acuerdo casi perfecto. Lo que no es reproducible es la
frontera entre primera y segunda calidad: 28 de los 35 desacuerdos están
ahí, y solo un caso de 100 se separó dos niveles. Ninguno de los dos
anotadores confundió fruta sana con podrida. Por eso el $\kappa$ ponderado
es tan superior al simple.

Esto además da el argumento para §V.C: el recall en `regular` no puede
superar de forma consistente el acuerdo humano en esa misma frontera, así
que parte del error residual es de la definición de la categoría y no del
modelo.

### 4.2 El tamaño en fotos propias es mucho peor de lo que dice el informe

§IV.C y §V.D citan exactitud de 0,8197 en fotografías propias. Pero el F1
macro ahí es **0,5349** y la exactitud balanceada **0,5288**: apenas por
encima del azar en un problema de tres clases. La exactitud alta viene del
desbalance de la partición.

Reportar solo exactitud justo en el único punto donde esa métrica engaña,
después de haber argumentado en §II.D y §III.A que F1 macro es la métrica
principal, es la inconsistencia más fácil de señalar en la sustentación.
Hay que decir el número malo y explicarlo: la etiqueta de tamaño se deriva
del diámetro relativo medido sobre la máscara, y en las fotos propias la
máscara es menos estable (fondos reales, manos, sombras), de modo que se
degrada la etiqueta y la característica a la vez.

### 4.3 La parada temprana nunca se activó en la CNN principal

`cnn_metrics.json` registra `mejor_epoca = 30` sobre un máximo de 30: la CNN
todavía mejoraba cuando terminó el entrenamiento. Es una limitación concreta
que explica parte de la diferencia frente a HistGradientBoosting, y es mejor
declararla que atribuirlo solo al «conjunto limitado».

Por contraste, en la ablación con máscara la parada temprana sí operó
(`mejor_epoca = 3`): esa configuración se degradó desde el principio.

### 4.4 Referencias y citas

La sección VII está vacía, pero el texto cita [1] y [2]. Además siguen
pendientes las cuatro citas de `docs/04` §3.2: pérdidas poscosecha de frutas
en Colombia, norma técnica de clasificación aplicable, licencia exacta del
dataset de Kaggle, y dos o tres trabajos publicados sobre el dataset para la
comparación con la literatura que pide la rúbrica.

### 4.5 Problema de espacio

El borrador ya ocupa las 7 páginas del límite **con cuatro tablas vacías, sin
referencias y sin figuras**. Al completarlo se pasa. Conviene recortar ahora:

- §II (fundamentos teóricos) repite buena parte de §III. Las fórmulas de
  pérdida y de pesos de clase pueden quedarse en un solo sitio.
- §V.I («Conclusiones del análisis») duplica §VI.
- Las fórmulas quedaron sin renderizar y se leen partidas: `M(x,y)`, `I`,
  `λ_t = 0,5`, el diámetro equivalente y los umbrales de tamaño aparecen
  cortados o vacíos. En LaTeX esto se arregla, pero hay que revisarlo porque
  ahora mismo hay frases que no dicen nada: «se consideran pequeños los
  valores , medianos los valores y grandes los valores».
- Los dos marcadores `[COMPLEMENTAR]` de §V.H siguen en el texto.
