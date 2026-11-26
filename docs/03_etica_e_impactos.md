# Análisis ético y de impactos

> Sustenta los criterios 1 (*Identificación y análisis de dilemas éticos*) y 2
> (*Análisis de impactos globales, sociales, ambientales y económicos*) de la
> rúbrica. Entre los dos pesan una cuarta parte de la evaluación.
>
> Marco de referencia: *ACM Code of Ethics and Professional Conduct* (2018) y
> *IEEE Code of Ethics*. Las citas entre corchetes remiten a los principios
> del código de la ACM.

---

## 1. Dilemas éticos del proyecto

Esta tabla no es genérica: cada fila corresponde a una situación concreta de
**este** proyecto, con la evidencia en el repositorio y la decisión adoptada.

| # | Dilema | Situación concreta | Principio | Decisión adoptada | Evidencia |
|---|---|---|---|---|---|
| 1 | **Privacidad de terceros** | Las 1.019 fotografías propias se tomaron en plazas de mercado, supermercados y hogares. Pueden contener manos, rostros al fondo, letreros de locales o identificadores del puesto | 1.6 Respetar la privacidad | Revisión automática con detector facial sobre las 1.019 imágenes: 39 marcadas, **0 rostros confirmados**. Las manos que sostienen la fruta no identifican a nadie. Queda pendiente la revisión de letreros | `src/data/revision_privacidad.py` |
| 2 | **Consentimiento del comerciante** | Se fotografió mercancía ajena dentro de establecimientos, posiblemente sin pedir permiso | 1.6, 2.3 | Documentar en el informe si se pidió autorización. Si no, no publicar imágenes donde el puesto sea identificable | — |
| 3 | **Sesgo de origen de los datos** | El 74 % de las imágenes viene de un dataset público capturado en otro país, con variedades, calibres y estándares de madurez distintos a los colombianos | 1.4 Ser justo y no discriminar | Medir explícitamente el desempeño por fuente (`kaggle` vs `recolectado`) en lugar de reportar un único número global | `src/evaluation/evaluate.py::analisis_por_fuente` |
| 4 | **Desempeño desigual entre clases** | El recall de `regular` es 0,585 frente a 0,95 de `good` y `bad`. El sistema falla justo en la categoría que decide si una fruta se vende con descuento o se descarta | 1.4, 2.5 | Reportar siempre el recall por clase y nunca solo la exactitud. Ponderación de clases en el entrenamiento. Recolectar más imágenes de la clase débil | `experiments/results/classic_quality_metrics.json` |
| 5 | **Asimetría de costos no declarada** | Un falso `good` manda fruta deteriorada al consumidor; un falso `bad` solo cuesta margen. Optimizar la exactitud trata ambos errores como equivalentes | 1.2 Evitar el daño | Matriz de costos explícita ($C_{bad \to good} = 5$) y reporte del riesgo empírico junto a las métricas estadísticas | `src/evaluation/metrics.py::costo_asimetrico` |
| 6 | **Etiquetas subjetivas presentadas como objetivas** | `regular` no tiene definición física: es un acuerdo entre cuatro estudiantes. El modelo aprende ese criterio particular y lo presenta como una medición | 1.3 Ser honesto y confiable | Protocolo de anotación escrito y público, con criterios observables y porcentajes de superficie afectada. Medición del acuerdo entre anotadores ($\kappa$ de Cohen) | `docs/01_metodologia_crisp_dm.md` §3.2 |
| 7 | **Riesgo de automatizar una decisión económica** | Si el sistema clasifica un lote como `bad`, alguien pierde dinero. El error del modelo tiene víctima concreta, normalmente el productor, que es el eslabón más débil | 1.2, 2.5 | El sistema se declara **de asistencia**, no de decisión. Debe mostrar la confianza y permitir que el operario anule la predicción. Toda predicción queda registrada y es auditable | Requisito de diseño de la GUI, `docs/01` §6.2 |
| 8 | **Desplazamiento laboral** | La clasificación manual es empleo real, con frecuencia informal y de baja calificación | 1.1 Contribuir al bienestar | Plantear el sistema como apoyo que reduce la carga repetitiva, no como reemplazo. Discutirlo abiertamente en el informe en lugar de omitirlo | §2.1 de este documento |
| 9 | **Uso indebido del sistema** | Un comprador grande podría usarlo para castigar sistemáticamente el precio al productor, amparándose en una supuesta objetividad técnica | 1.2, 2.5 | Documentar la incertidumbre del modelo. Recomendar que los umbrales de clasificación sean acordados entre las partes, no fijados unilateralmente | — |
| 10 | **Propiedad y licencia de los datos** | Se usa un dataset público de Kaggle cuya licencia hay que respetar y citar | 1.5 Respetar el trabajo ajeno | Citar la fuente en el README y en el informe. **Pendiente: verificar y registrar la licencia exacta del dataset** | `README.md` |
| 11 | **Transparencia sobre las limitaciones** | La estimación de tamaño no mide tamaño físico (es ambigua respecto a la distancia de la cámara), y los modelos clásicos alcanzan $F_1 = 1{,}000$ por circularidad del diseño | 1.3 | Declararlo explícitamente en el informe en vez de presentar el 1,000 como logro | `docs/02_formulacion_matematica.md` §5 |
| 12 | **Reproducibilidad y honestidad en los resultados** | Con un 37 % de imágenes casi duplicadas, una partición aleatoria produce métricas infladas que parecerían excelentes | 1.3, 2.1 Alcanzar la máxima calidad | Partición por grupos de duplicados y publicación del código completo que la genera | `src/features/build_features.py` |

### 1.1 Dos dilemas que seguimos teniendo abiertos

1. **Compartir el dataset entre grupos.** El enunciado pide un mecanismo común
   de intercambio de anotaciones. Antes de subir nada, hay que resolver el
   dilema 1: las fotografías pueden contener personas identificables. Nuestra
   posición: no se comparte ninguna imagen sin revisión previa.
2. **El modelo aprende el criterio de cuatro estudiantes, no un estándar de la
   industria.** Mientras no se contraste con la norma técnica de calidad de
   frutas aplicable (ICONTEC o la clasificación comercial del mercado local),
   el sistema reproduce una convención privada. Conviene decirlo en el informe
   y proponerlo como trabajo futuro.

---

## 2. Análisis de impactos

### 2.1 Impacto social

**Positivos**
- Reduce la arbitrariedad: el mismo criterio a las 6 a. m. y a las 6 p. m., para
  el proveedor conocido y para el desconocido.
- Da al pequeño productor un argumento verificable frente al comprador, en
  lugar de depender del ojo del intermediario.
- Disminuye la carga de una tarea repetitiva y visualmente agotadora.

**Negativos**
- Desplazamiento de empleo de clasificación manual, frecuentemente informal.
- Dependencia tecnológica: si el sistema falla o no hay energía, el proceso se
  detiene si se eliminó la capacidad manual.
- Brecha digital: quien no tenga celular con cámara decente queda fuera.
- **Falsa autoridad.** El riesgo más sutil: una cifra en pantalla se percibe
  como objetiva y cierra la discusión, aunque sea una predicción con un 8,5 %
  de error y un recall de 0,585 en la clase intermedia.

**Mitigación:** sistema de asistencia con confianza visible y anulación
manual; capacitación del operario en lugar de reemplazo; registro auditable de
las predicciones.

### 2.2 Impacto económico

**Positivos**
- Menos devoluciones por fruta deteriorada enviada al cliente.
- Estandarización que permite acordar precios por categoría y reduce la
  disputa en la negociación.
- Costo de implementación bajo: corre en CPU y con un celular, sin hardware
  especializado.

**Negativos**
- Costo inicial de montaje y capacitación, difícil de asumir para un pequeño
  comerciante.
- Un error sistemático sesgado hacia `bad` castigaría de forma persistente a
  los productores y concentraría el beneficio en el comprador.
- Sobreajuste al dominio: un modelo entrenado mayormente con imágenes de
  estudio puede rendir peor en una bodega real, y esa caída es invisible si no
  se mide por fuente.

**Mitigación:** medir el desempeño por dominio antes de desplegar; definir los
umbrales de forma acordada entre las partes; auditoría periódica de la tasa de
error por categoría.

### 2.3 Impacto ambiental

**Positivos**
- Menos desperdicio alimentario: separar la fruta `regular` en lugar de
  descartar el lote completo redirige producto al procesamiento (jugos,
  pulpas, compotas) en vez de al relleno sanitario.
- Menos transporte inútil de fruta que se va a rechazar en destino.
- Menos emisiones de metano asociadas a la descomposición en vertedero.

**Negativos**
- Huella de cómputo del entrenamiento. En este proyecto es modesta y
  cuantificable: la CNN tiene 304.742 parámetros y se entrenó en CPU, del
  orden de una hora. Es una decisión deliberada frente a usar una red grande
  preentrenada.
- Ciclo de vida del hardware (celulares, cámaras) si el despliegue exige
  equipos nuevos.

**Mitigación:** reutilizar equipos existentes; reentrenar solo cuando las
métricas lo justifiquen, no por costumbre; reportar el costo de cómputo en el
informe.

### 2.4 Impacto global

- **Transferible, pero no directamente.** La arquitectura sirve para otras
  cadenas agroindustriales, pero los criterios de calidad son locales: lo que
  es `regular` en Cali no lo es en otro mercado. Un modelo entrenado aquí no
  debe desplegarse en otro país sin reanotar.
- **Seguridad alimentaria.** Reducir pérdidas poscosecha es una de las palancas
  más directas sobre la disponibilidad de alimento, sin ampliar la frontera
  agrícola.
- **Asimetría de poder.** Quien controla el modelo controla el criterio de
  clasificación. Si la herramienta queda solo en manos de los grandes
  compradores, amplifica una asimetría que ya existe. Que el código sea
  abierto y el criterio público es, en sí mismo, una decisión ética.

### 2.5 Matriz de impactos

| Dimensión | Impacto positivo principal | Riesgo principal | Severidad | Mitigación | ¿Implementada? |
|---|---|---|---|---|---|
| Social | Criterio consistente y auditable | Falsa autoridad de la cifra | Alta | Mostrar confianza y permitir anulación | Diseño de la GUI, pendiente |
| Social | Menos carga repetitiva | Desplazamiento laboral | Media | Enfoque de asistencia, no reemplazo | Declarado |
| Económico | Menos devoluciones y disputas | Sesgo sistemático contra el productor | Alta | Umbrales acordados y auditoría de error por clase | Métricas por clase, sí |
| Económico | Costo de implementación bajo | Caída de desempeño en el dominio real | Alta | Evaluación por fuente antes de desplegar | Implementada |
| Ambiental | Menos desperdicio alimentario | Huella de cómputo | Baja | Modelo pequeño entrenado en CPU | Implementada |
| Global | Transferible a otras cadenas | Criterios locales no transferibles | Media | Exigir reanotación por mercado | Documentado |
| Global | Aporta a la seguridad alimentaria | Concentración del criterio en pocos actores | Media | Código y criterio públicos | Implementada |

---

## 3. Compromisos verificables del grupo

Lo que un lector del informe puede comprobar en el repositorio:

1. El criterio de clasificación está escrito y es público (`docs/01` §3.2).
2. Las métricas se reportan por clase y por fuente, no solo en agregado.
3. Los errores costosos se cuentan por separado (falsos `good` sobre `bad`).
4. Toda corrección de etiqueta queda registrada en
   `docs/anotacion/correcciones_etiquetas.csv` con su motivo y su revisor, y
   se aplica automáticamente al reconstruir el dataset. Reetiquetar sin dejar
   rastro sería indistinguible de ajustar los datos para mejorar las
   métricas.
5. Las limitaciones conocidas están documentadas, incluida la que vuelve
   trivial la tarea de tamaño.
6. El código que genera cada número es reproducible con semilla fija.

## 4. Pendientes éticos antes de la entrega

- [x] **Revisar las fotografías propias por rostros.** Hecho: el detector
      facial YuNet recorrió las 1.019 fotografías
      (`python -m src.data.revision_privacidad`). Marcó 39 imágenes y la
      revisión humana confirmó que **ninguna contiene un rostro**: son falsos
      positivos sobre manzanas con manchas, donde el patrón de dos motas
      oscuras y una tercera debajo activa el detector, más una fotografía con
      dedos en el encuadre. Informe en
      `docs/anotacion/revision_privacidad.csv`.
- [ ] Revisar a ojo letreros de establecimientos, documentos y etiquetas de
      precio identificables; el detector automático no los cubre.
- [ ] Documentar si se pidió autorización a los comerciantes.
- [ ] Verificar y citar la licencia exacta del dataset de Kaggle.
- [ ] Medir el $\kappa$ de Cohen entre dos anotadores sobre 100 imágenes
      comunes y reportarlo.
- [ ] Decidir y documentar las condiciones bajo las cuales se comparte el
      dataset con los demás grupos del curso.

---

## Referencias

- Association for Computing Machinery, *ACM Code of Ethics and Professional
  Conduct*, 2018. [En línea]. Disponible: https://www.acm.org/code-of-ethics
- Institute of Electrical and Electronics Engineers, *IEEE Code of Ethics*.
  [En línea]. Disponible: https://www.ieee.org/about/corporate/governance/p7-8.html

> **[CITAS PENDIENTES]** Añadir en formato IEEE: (a) una fuente sobre pérdidas
> poscosecha de frutas en Colombia (FAO, DANE o DNP); (b) la norma técnica de
> clasificación de calidad de frutas aplicable; (c) la licencia del dataset de
> Kaggle. La rúbrica exige referencias IEEE de material efectivamente leído.
