# Formulación matemática del problema

> Este documento sustenta el criterio 3 de la rúbrica (*Formulación matemática
> del problema*) y el criterio 5 (*Aplicación rigurosa de fundamentos
> matemáticos*). Cada ecuación aquí descrita corresponde a código concreto del
> repositorio; la referencia al archivo aparece al final de cada sección.

---

## 1. Planteamiento formal

### 1.1 Espacios

Una fotografía es un tensor de intensidades

$$
I \in \mathcal{X} = \{0,1,\dots,255\}^{H \times W \times 3}
$$

Para una imagen de celular típica de este proyecto ($H \times W = 3024 \times 4032$)
el espacio de entrada tiene $3024 \cdot 4032 \cdot 3 \approx 3{,}66 \times 10^{7}$
dimensiones y cardinalidad $256^{3{,}66\times 10^{7}}$. El sistema debe aprender
una función sobre ese espacio a partir de $N = 3{.}356$ ejemplos etiquetados.
Esta desproporción entre dimensionalidad y número de muestras es la primera
razón por la que el problema es complejo en el sentido ingenieril: **no se
puede resolver por tabulación ni por reglas explícitas**, y cualquier método
que no reduzca la dimensionalidad de forma informada sobreajusta.

Las salidas son dos variables categóricas:

$$
\mathcal{Y}_q = \{\text{good}, \text{regular}, \text{bad}\}, \qquad
\mathcal{Y}_s = \{\text{pequeño}, \text{mediano}, \text{grande}\}
$$

$\mathcal{Y}_s$ es **ordinal**: existe un orden total
$\text{pequeño} \prec \text{mediano} \prec \text{grande}$, y
$\mathcal{Y}_q$ también lo es si se interpreta como grado de deterioro. Este
hecho cambia qué métricas son válidas (sección 7.3).

### 1.2 El problema de aprendizaje

Se busca un par de funciones

$$
h_q : \mathcal{X} \to \mathcal{Y}_q, \qquad h_s : \mathcal{X} \to \mathcal{Y}_s
$$

que minimicen el **riesgo esperado** bajo la distribución desconocida $\mathcal{D}$
de la que provienen las frutas que llegarán a la bodega:

$$
R(h) = \mathbb{E}_{(I,y)\sim\mathcal{D}}\big[ L(y, h(I)) \big]
$$

Como $\mathcal{D}$ es inaccesible, se minimiza el **riesgo empírico regularizado**
sobre la muestra de entrenamiento $S = \{(I_n, y_n)\}_{n=1}^{N}$:

$$
\hat{h} = \arg\min_{h \in \mathcal{H}} \; \frac{1}{N}\sum_{n=1}^{N} L\big(y_n, h(I_n)\big) + \lambda\,\Omega(h)
$$

donde $\mathcal{H}$ es la clase de hipótesis (regresión logística, SVM, bosque,
CNN), $\Omega$ el término de regularización y $\lambda$ su intensidad. **Todo el
proyecto consiste en elegir $\mathcal{H}$, $L$, $\Omega$ y $\lambda$ de forma
justificada.**

### 1.3 Por qué es un problema complejo de ingeniería

| Fuente de dificultad | Manifestación concreta en este proyecto |
|---|---|
| Alta dimensionalidad | $3{,}7\times10^{7}$ variables de entrada por $3{.}356$ muestras |
| Problema mal planteado | La proyección 3D→2D pierde la escala: dos frutas de tamaño distinto producen la misma imagen a distancias distintas (sección 5) |
| Varianza intraclase | Una manzana "buena" varía en color, iluminación, orientación y fondo |
| Frontera difusa | "regular" no tiene definición física; es un acuerdo de anotación |
| Ruido de etiqueta | Anotación humana, subjetiva, por cinco personas distintas |
| Cambio de dominio | Entrenar con Kaggle (estudio) y desplegar en plaza de mercado |
| Desbalance | $N_{\text{regular}}/N = 221/3356 = 6{,}6\,\%$ |
| Costos asimétricos | Vender fruta podrida cuesta mucho más que descartar fruta buena |
| Restricción de cómputo | Debe correr en CPU y responder en tiempo cuasi-real en la GUI |

---

## 2. Reducción del problema: segmentación

En lugar de aprender directamente sobre $\mathcal{X}$, se descompone el sistema:

$$
h = g \circ \varphi \circ \sigma
$$

donde $\sigma$ segmenta la fruta, $\varphi$ extrae características y $g$
clasifica. Esta factorización es la decisión de diseño central del proyecto:
reduce la dimensionalidad de $3{,}7\times10^{7}$ a $93$ **usando conocimiento
del dominio en lugar de datos**, que es exactamente lo que se necesita cuando
las muestras son escasas.

### 2.1 Umbralización de Otsu sobre saturación

Sea $S(x,y)$ el canal de saturación en HSV. Otsu elige el umbral $t^\*$ que
maximiza la varianza entre clases del histograma:

$$
t^{*} = \arg\max_{t} \;\sigma_B^2(t), \qquad
\sigma_B^2(t) = \omega_0(t)\,\omega_1(t)\,\big[\mu_0(t)-\mu_1(t)\big]^2
$$

con $\omega_0(t) = \sum_{i=0}^{t} p_i$ la masa de probabilidad del fondo,
$\omega_1 = 1-\omega_0$, y $\mu_0,\mu_1$ las medias de cada grupo. Maximizar
$\sigma_B^2$ es equivalente a minimizar la varianza intraclase
$\sigma_W^2 = \sigma^2 - \sigma_B^2$, porque la varianza total $\sigma^2$ es
constante.

**Por qué saturación y no intensidad:** las frutas son cromáticas
($S$ alta) y los fondos del proyecto son mesones, papel o manos
(aproximadamente acromáticos, $S$ baja). La iluminación afecta sobre todo a
$V$, no a $S$, de modo que umbralizar $S$ es más robusto a las sombras. Se
impone además un piso $t = \max(t^{*}, 45)$ porque cuando la imagen es casi
toda fruta, Otsu parte la fruta misma en dos.

### 2.2 Morfología matemática

La máscara cruda se limpia con operaciones de conjuntos. Con $B$ un elemento
estructurante elíptico de $9\times9$:

$$
\text{apertura: } M \circ B = (M \ominus B) \oplus B, \qquad
\text{cierre: } M \bullet B = (M \oplus B) \ominus B
$$

donde $\oplus$ es la dilatación y $\ominus$ la erosión. El cierre rellena
agujeros internos (brillos especulares que el umbral marcó como fondo) y la
apertura elimina motas aisladas (textura del granito del mesón). El orden
importa: cierre primero evita fragmentar la fruta antes de limpiar el ruido.

### 2.3 Selección del contorno

De los contornos resultantes $\{C_k\}$ se elige la fruta central maximizando

$$
k^{*} = \arg\max_k \; A(C_k)\left(1 - 0{,}7\,\frac{\lVert \mathbf{c}_k - \mathbf{c}_{\text{img}}\rVert_2}{\sqrt{H^2+W^2}}\right)
$$

con $A$ el área y $\mathbf{c}_k = \big(\tfrac{m_{10}}{m_{00}}, \tfrac{m_{01}}{m_{00}}\big)$
el centroide calculado con los momentos espaciales
$m_{pq} = \sum_{x}\sum_{y} x^p y^q M(x,y)$. El factor penaliza objetos
periféricos (una mano, otra fruta al borde) sin descartarlos del todo.

*Código: `src/data/build_dataset.py::get_fruit_mask_and_size`*

---

## 3. Mapa de características $\varphi$

El vector $\varphi(I) \in \mathbb{R}^{93}$ se compone de cinco bloques. Todos
se calculan **solo sobre los píxeles de la máscara**
$\mathcal{M} = \{(x,y) : M(x,y) > 0\}$, con $N_\mathcal{M} = |\mathcal{M}|$.

### 3.1 Histogramas marginales de color (64 dimensiones)

Para cada canal $c \in \{H, S, V\}$ con $B_c$ bins:

$$
p^{(c)}_k = \frac{1}{N_\mathcal{M}} \sum_{(x,y)\in\mathcal{M}} \mathbb{1}\big[ \text{bin}_k \le c(x,y) < \text{bin}_{k+1} \big],
\qquad \sum_{k=1}^{B_c} p^{(c)}_k = 1
$$

Es la **distribución empírica** del color de la fruta. La normalización por
$N_\mathcal{M}$ hace el descriptor invariante al número de píxeles, es decir a
la distancia de la cámara.

**Por qué marginales y no conjunto:** un histograma 3D de $16\times8\times8$
tiene $1.024$ bins y, con $N_\mathcal{M} \approx 10^5$ píxeles, casi todos
quedan vacíos o con conteos de un dígito: el estimador tiene varianza enorme.
Las marginales con $32+16+16 = 64$ bins estiman tres distribuciones de baja
dimensión con muchos más píxeles por bin. Se paga el precio de perder la
correlación entre canales, que en este problema aportaba poco.

### 3.2 Momentos de color (9 dimensiones)

$$
\mu_c = \frac{1}{N_\mathcal{M}}\sum_{\mathcal{M}} c, \qquad
\sigma_c = \sqrt{\frac{1}{N_\mathcal{M}}\sum_{\mathcal{M}} (c-\mu_c)^2}, \qquad
\gamma_c = \frac{1}{N_\mathcal{M}}\sum_{\mathcal{M}}\left(\frac{c-\mu_c}{\sigma_c}\right)^3
$$

La asimetría $\gamma_V$ es el descriptor más informativo de este bloque: una
fruta con manchas oscuras localizadas produce una cola izquierda larga
($\gamma_V < 0$) aunque su brillo medio $\mu_V$ sea normal.

### 3.3 Descriptores de forma (6 dimensiones)

Con $A$ el área, $P$ el perímetro, $(w,h)$ la caja envolvente y
$A_{\text{conv}}$ el área del casco convexo:

$$
d_{\text{rel}} = \frac{1}{\max(H,W)}\sqrt{\frac{4A}{\pi}}, \qquad
\text{razón de aspecto} = \frac{w}{h}, \qquad
\text{extensión} = \frac{A}{wh}
$$

$$
\text{solidez} = \frac{A}{A_{\text{conv}}}, \qquad
\text{circularidad} = \frac{4\pi A}{P^2}, \qquad
e = \sqrt{1-\left(\frac{b}{a}\right)^{\!2}}
$$

El **diámetro equivalente** $\sqrt{4A/\pi}$ invierte $A = \pi d^2/4$: es el
diámetro del círculo de igual área. La **circularidad** es el cociente
isoperimétrico, que por la desigualdad isoperimétrica $4\pi A \le P^2$ cumple
$0 < 4\pi A/P^2 \le 1$ con igualdad solo en el círculo; distingue limones de
bananos. La **solidez** $< 1$ detecta concavidades: mordeduras, cortes y
deformaciones por golpe. $e$ es la excentricidad de la elipse ajustada por
mínimos cuadrados, con semiejes $a \ge b$.

### 3.4 Textura: patrones binarios locales (10 dimensiones)

Para cada píxel $(x,y)$ y sus $P=8$ vecinos a radio $R=1$:

$$
\text{LBP}_{P,R}(x,y) = \sum_{i=0}^{P-1} \mathbb{1}\big[\, g_i \ge g_c \,\big]\, 2^{i}
$$

con $g_c$ la intensidad del centro y $g_i$ la del vecino $i$. El código crudo
tiene $2^8 = 256$ valores. Se colapsa con la variante rotación-invariante
uniforme (riu2), definida mediante la medida de uniformidad

$$
U(\text{patrón}) = \big| \mathbb{1}[g_{P-1}\ge g_c] - \mathbb{1}[g_0 \ge g_c] \big|
+ \sum_{i=1}^{P-1} \big| \mathbb{1}[g_i \ge g_c] - \mathbb{1}[g_{i-1}\ge g_c] \big|
$$

$$
\text{LBP}^{riu2}_{P,R} =
\begin{cases}
\displaystyle\sum_{i=0}^{P-1}\mathbb{1}[g_i \ge g_c] & \text{si } U \le 2 \\[2mm]
P+1 = 9 & \text{en otro caso}
\end{cases}
$$

Esto reduce 256 códigos a 10 etiquetas, **invariantes a rotación** (porque solo
cuentan el número de unos, no su posición) y **a cambios monótonos de
iluminación** (porque solo usan comparaciones $\ge$, no diferencias). El
descriptor final es el histograma normalizado de las 10 etiquetas.

Referencia: Ojala, Pietikäinen y Mäenpää, *IEEE TPAMI* 24(7), 2002.

### 3.5 Indicadores de defecto (4 dimensiones)

$$
\text{frac}_{\text{oscuros}} = \frac{1}{N_\mathcal{M}}\big|\{V < 0{,}5\,\mathrm{med}(V)\}\big|, \qquad
\text{frac}_{\text{pardos}} = \frac{1}{N_\mathcal{M}}\big|\{10 \le H \le 25 \;\wedge\; S < 150\}\big|
$$

$$
\text{contraste} = \frac{1}{255}\,\mathrm{sd}\!\left(\sqrt{G_x^2+G_y^2}\right), \qquad
\mathcal{H}(V) = -\frac{1}{\log_2 32}\sum_{k=1}^{32} p_k \log_2 p_k
$$

donde $G_x, G_y$ son las derivadas de Sobel. El umbral relativo a la mediana
(no absoluto) hace que $\text{frac}_{\text{oscuros}}$ sea invariante a la
exposición global de la foto: detecta manchas *relativas* a la propia fruta. La
entropía normalizada mide la heterogeneidad superficial: una fruta sana es
homogénea ($\mathcal{H}$ baja) y una podrida es heterogénea.

*Código: `src/features/extractors.py`*

---

## 4. Modelos y sus funciones objetivo

### 4.1 Regresión logística multinomial

$$
P(y=k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^\top \mathbf{x} + b_k)}{\sum_{j=1}^{K}\exp(\mathbf{w}_j^\top \mathbf{x} + b_j)}
$$

$$
\min_{W,b} \; -\sum_{n=1}^{N} \alpha_{y_n} \log P(y_n \mid \mathbf{x}_n) \;+\; \frac{1}{2C}\lVert W \rVert_F^2
$$

El hiperparámetro $C$ es el inverso de la intensidad de regularización; se
ajusta en $\{10^{-2},\dots,10^{2}\}$.

### 4.2 Máquina de soporte vectorial con núcleo gaussiano

Problema primal con márgenes blandos:

$$
\min_{\mathbf{w},b,\xi} \; \frac{1}{2}\lVert \mathbf{w}\rVert^2 + C\sum_{n} \alpha_{y_n}\xi_n
\quad \text{s.a.} \quad y_n(\mathbf{w}^\top \phi(\mathbf{x}_n)+b) \ge 1-\xi_n, \; \xi_n \ge 0
$$

Por el teorema del representante no hace falta conocer $\phi$, solo el núcleo

$$
k(\mathbf{x},\mathbf{x}') = \exp\!\big(-\gamma \lVert \mathbf{x}-\mathbf{x}'\rVert^2\big)
$$

que corresponde a un espacio de características de dimensión infinita. $\gamma$
controla el radio de influencia de cada muestra: con $\gamma$ grande la
frontera se vuelve muy local y el modelo memoriza. La opción `scale` usa
$\gamma = 1/(d\,\mathrm{Var}(X))$.

**Por qué el escalado es obligatorio aquí:** el núcleo depende de la distancia
euclidiana, así que una característica con rango $[0,255]$ domina por completo
a otra con rango $[0,1]$. Se estandariza $z = (x-\mu)/\sigma$ **con la media y
desviación calculadas solo en el pliegue de entrenamiento**, dentro de un
`Pipeline`, para no filtrar información del conjunto de prueba.

### 4.3 Random Forest

Cada nodo parte el espacio minimizando la impureza de Gini ponderada:

$$
G(\mathcal{S}) = 1 - \sum_{k=1}^{K} \hat{p}_k^2, \qquad
\Delta G = G(\mathcal{S}) - \frac{|\mathcal{S}_L|}{|\mathcal{S}|}G(\mathcal{S}_L) - \frac{|\mathcal{S}_R|}{|\mathcal{S}|}G(\mathcal{S}_R)
$$

La predicción es el voto mayoritario de $T$ árboles, cada uno entrenado sobre
una muestra bootstrap y considerando $\sqrt{d}$ características por nodo. La
varianza del promedio de $T$ predictores con correlación $\rho$ es

$$
\mathrm{Var} = \rho\,\sigma^2 + \frac{1-\rho}{T}\sigma^2
$$

lo que explica por qué el muestreo aleatorio de características (que reduce
$\rho$) importa más que aumentar $T$ indefinidamente.

### 4.4 Gradient boosting por histogramas

Modelo aditivo $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu\, t_m(\mathbf{x})$,
donde $t_m$ se ajusta al gradiente negativo de la pérdida respecto a la
predicción actual:

$$
t_m \approx \arg\min_{t}\sum_n \left( -\left[\frac{\partial L(y_n, F)}{\partial F}\right]_{F=F_{m-1}} - t(\mathbf{x}_n)\right)^2
$$

Es descenso de gradiente en el espacio de funciones. $\nu$ (tasa de
aprendizaje) y el número de iteraciones $M$ se compensan mutuamente: menor
$\nu$ exige mayor $M$ pero generaliza mejor.

*Código: `src/models/classic.py`, `src/training/train_classic.py`*

---

## 5. Estimación de tamaño: el punto débil del diseño actual

### 5.1 Modelo de cámara estenopeica

Bajo proyección perspectiva, un objeto de diámetro físico $D$ a distancia $Z$
del sensor, con distancia focal $f$ (en píxeles), proyecta

$$
d_{\text{px}} = f\,\frac{D}{Z} \quad \Longrightarrow \quad D = \frac{d_{\text{px}}\,Z}{f}
$$

### 5.2 Por qué $d_{\text{rel}}$ no mide tamaño

El descriptor implementado es

$$
d_{\text{rel}} = \frac{d_{\text{px}}}{\max(H,W)} = \frac{f}{\max(H,W)}\cdot\frac{D}{Z}
$$

que depende del **cociente** $D/Z$. Una naranja pequeña a 20 cm y una grande a
40 cm producen el mismo $d_{\text{rel}}$. Es decir, **el problema está mal
planteado**: sin información adicional, el tamaño físico no es observable a
partir de una sola imagen. Formalmente, la función de proyección no es
inyectiva respecto a $D$.

### 5.3 Consecuencia experimental observada

Las etiquetas de tamaño se generan discretizando $d_{\text{rel}}$:

$$
y_s =
\begin{cases}
\text{pequeño} & d_{\text{rel}} < 0{,}35\\
\text{mediano} & 0{,}35 \le d_{\text{rel}} \le 0{,}58\\
\text{grande} & d_{\text{rel}} > 0{,}58
\end{cases}
$$

y $d_{\text{rel}}$ es a la vez **una de las 93 características de entrada**. El
modelo no aprende nada sobre tamaño: aprende a reproducir una función escalón
de una de sus propias entradas. Eso explica el $F_1 = 1{,}000$ obtenido por
Random Forest y por gradient boosting, que **no es un buen resultado sino un
diagnóstico de circularidad en el diseño del experimento**.

### 5.4 Soluciones válidas

1. **Objeto de referencia en la escena.** Una moneda de diámetro conocido
   $D_{\text{ref}}$ da la escala directamente:
   $D = d_{\text{px}} \cdot D_{\text{ref}} / d_{\text{px,ref}}$, eliminando $f$ y $Z$.
2. **Protocolo de captura a distancia fija.** Si $Z$ y $f$ son constantes,
   $d_{\text{px}}$ es proporcional a $D$ y la discretización sí es válida. Exige
   documentar el montaje (trípode o soporte a altura fija).
3. **Anotación manual con medición física.** Pesar o medir con calibrador una
   submuestra y usarla como verdad de referencia.
4. Si no se implementa ninguna, **renombrar la salida** a "tamaño aparente en el
   encuadre" y declarar explícitamente la limitación.

> **Recomendación para el informe:** implementar la opción 2 para las
> fotografías propias y reportar la opción 4 para las imágenes de Kaggle, cuyo
> protocolo de captura desconocemos.

---

## 6. Red neuronal convolucional

### 6.1 Operador de convolución

$$
(\mathbf{X} * \mathbf{K})_{i,j,o} = \sum_{c=1}^{C_{in}}\sum_{u=-1}^{1}\sum_{v=-1}^{1} K_{u,v,c,o}\; X_{i+u,\,j+v,\,c} \;+\; b_o
$$

Dos propiedades justifican su uso frente a una capa densa:
**equivarianza a traslación** (una mancha detectada en una esquina se detecta
igual en el centro) y **compartición de parámetros** (un filtro $3\times3$ tiene
9 pesos por canal, independientemente de $H$ y $W$).

**Campo receptivo.** Dos convoluciones $3\times3$ apiladas tienen campo
receptivo efectivo $5\times5$ con $2\cdot 9 = 18$ pesos por canal frente a los
$25$ de una sola $5\times5$, y añaden una no linealidad intermedia. Tras tres
bloques con submuestreo $2\times$, el campo receptivo cubre
$\approx 44\times44$ píxeles sobre una entrada de $128\times128$.

### 6.2 Normalización por lotes

$$
\hat{x}^{(i)} = \frac{x^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad
y^{(i)} = \lambda \hat{x}^{(i)} + \beta
$$

con $\mu_B, \sigma_B^2$ la media y varianza del minilote y $\lambda, \beta$
parámetros aprendidos. Estabiliza la distribución de activaciones entre capas y
permite tasas de aprendizaje mayores.

### 6.3 Global Average Pooling

$$
z_o = \frac{1}{H'W'}\sum_{i=1}^{H'}\sum_{j=1}^{W'} A_{i,j,o}
$$

Colapsa el mapa final de $128\times16\times16 = 32{.}768$ activaciones a $128$
valores. Si se aplanara, la primera capa densa tendría
$32{.}768 \times 128 \approx 4{,}2$ millones de parámetros, más que el resto de la
red junta, y con $N = 2{.}300$ imágenes de entrenamiento el sobreajuste sería
inmediato. **Es la principal defensa estructural contra el sobreajuste.**

### 6.4 Pérdida multitarea

$$
\mathcal{L} = \lambda_q \underbrace{\left(-\sum_{k} \tilde{y}_k^{q}\, \alpha_k^q \log \hat{p}_k^{q}\right)}_{\text{entropía cruzada de calidad}}
\;+\; \lambda_s \left(-\sum_{k} \tilde{y}_k^{s}\, \alpha_k^s \log \hat{p}_k^{s}\right)
$$

con $\lambda_q = 1{,}0$ y $\lambda_s = 0{,}5$. El peso menor de la tarea de
tamaño refleja que sus etiquetas son menos confiables (sección 5).

**Suavizado de etiquetas.** En lugar del vector one-hot se usa

$$
\tilde{y}_k = (1-\varepsilon)\,\mathbb{1}[k = y] + \frac{\varepsilon}{K}, \qquad \varepsilon = 0{,}05
$$

lo que impide que la red lleve los logits a $\pm\infty$ y reduce el exceso de
confianza, especialmente importante aquí porque el 38 % de las imágenes tiene
casi duplicados.

**Pesos de clase.** Para compensar el desbalance:

$$
\alpha_k = \frac{N}{K\,N_k}
$$

Con $N_{\text{regular}} = 221$ frente a $N_{\text{good}} = 1652$, la clase
minoritaria recibe un peso $\approx 7{,}5$ veces mayor. Esto convierte la
minimización de la entropía cruzada en una aproximación a maximizar la
**exactitud balanceada** en lugar de la exactitud cruda.

### 6.5 Optimización

AdamW con momentos de primer y segundo orden:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\theta_t = \theta_{t-1} - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \omega\,\theta_{t-1}\right)
$$

El decaimiento de peso $\omega$ se aplica **desacoplado** del gradiente
adaptativo (ahí está la diferencia con Adam clásico), lo que lo convierte en
una regularización $L_2$ genuina. La tasa sigue un programa cosenoidal

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_0 - \eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

y los gradientes se recortan a $\lVert g \rVert_2 \le 5$ para evitar pasos
destructivos en las primeras épocas.

*Código: `src/models/cnn.py`, `src/training/train_cnn.py`*

---

## 7. Evaluación

### 7.1 Métricas de clasificación

Con $\mathrm{TP}_k, \mathrm{FP}_k, \mathrm{FN}_k$ los conteos de la clase $k$:

$$
\mathrm{P}_k = \frac{\mathrm{TP}_k}{\mathrm{TP}_k+\mathrm{FP}_k}, \qquad
\mathrm{R}_k = \frac{\mathrm{TP}_k}{\mathrm{TP}_k+\mathrm{FN}_k}, \qquad
F_{1,k} = \frac{2\,\mathrm{P}_k \mathrm{R}_k}{\mathrm{P}_k+\mathrm{R}_k}
$$

$$
F_1^{\text{macro}} = \frac{1}{K}\sum_{k=1}^{K} F_{1,k}, \qquad
\text{BA} = \frac{1}{K}\sum_{k=1}^{K} \mathrm{R}_k
$$

El promedio **macro** es el correcto aquí: da el mismo peso a "regular"
(6,6 % de los datos) que a "good" (49 %). Un promedio ponderado permitiría a un
modelo ignorar la clase minoritaria y seguir luciendo bien.

$$
\kappa = \frac{p_o - p_e}{1-p_e}, \qquad
\mathrm{MCC} = \frac{\mathrm{TP}\cdot\mathrm{TN}-\mathrm{FP}\cdot\mathrm{FN}}{\sqrt{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})(\mathrm{TN}+\mathrm{FP})(\mathrm{TN}+\mathrm{FN})}}
$$

$\kappa$ descuenta el acuerdo esperado por azar $p_e$; MCC es la correlación de
Pearson entre predicción y verdad tratadas como variables binarias, y solo es
alta si las cuatro celdas de la matriz son buenas.

### 7.2 Decisión bayesiana con costos asimétricos

El error de vender fruta podrida no cuesta lo mismo que descartar fruta buena.
Con una matriz de costos $C_{ij}$ (costo de predecir $j$ cuando la verdad es
$i$):

$$
C = \begin{pmatrix} 0 & 1 & 1\\ 1 & 0 & 1\\ 5 & 1 & 0 \end{pmatrix}
\qquad (\text{orden: good, regular, bad})
$$

el riesgo empírico es

$$
\hat{R} = \frac{1}{N}\sum_{n=1}^{N} C_{y_n,\hat{y}_n}
$$

y la **regla de decisión óptima de Bayes** no es el argmax de la probabilidad
sino el mínimo del riesgo condicional:

$$
\hat{y}(\mathbf{x}) = \arg\min_{j} \sum_{i} C_{ij}\, P(y=i \mid \mathbf{x})
$$

Con $C_{\text{bad}\to\text{good}} = 5$, la regla exige
$P(\text{bad}\mid\mathbf{x}) < 1/5$ antes de declarar una fruta como buena. Esto
permite justificar la selección del modelo en términos del negocio y no solo
estadísticos.

### 7.3 Métrica ordinal para tamaño

Codificando $\text{pequeño}=0$, $\text{mediano}=1$, $\text{grande}=2$:

$$
\mathrm{MAE}_{\text{ord}} = \frac{1}{N}\sum_{n=1}^{N} \lvert y_n - \hat{y}_n \rvert
$$

Confundir pequeño con grande ($|0-2| = 2$) cuesta el doble que confundir
pequeño con mediano ($|0-1| = 1$). Ninguna métrica nominal captura esto.

*Código: `src/evaluation/metrics.py`*

---

## 8. Validación: por qué la partición aleatoria está mal

### 8.1 El supuesto que se rompe

Toda la teoría del riesgo empírico asume muestras **i.i.d.** El dataset de
Kaggle contiene múltiples fotografías casi idénticas de la misma fruta
(recortes, rotaciones y ajustes de brillo del mismo original). Si copias del
mismo objeto caen simultáneamente en entrenamiento y en prueba, el conjunto de
prueba deja de ser independiente y $\hat{R}_{\text{test}}$ ya no estima
$R(h)$: estima la capacidad de memorización.

### 8.2 Detección de casi duplicados

Se calcula una huella perceptual dHash de 64 bits comparando cada píxel con su
vecino derecho en una miniatura de $9\times8$ en escala de grises:

$$
b_{ij} = \mathbb{1}\big[\, g(i,j) > g(i,j+1) \,\big], \qquad
\mathcal{h}(I) = \sum_{i=0}^{7}\sum_{j=0}^{7} b_{ij}\,2^{8i+j}
$$

Dos imágenes se consideran la misma fruta si su distancia de Hamming

$$
d_H(\mathcal{h}_1, \mathcal{h}_2) = \lVert \mathcal{h}_1 \oplus \mathcal{h}_2 \rVert_1 \le 6
$$

La relación "ser casi duplicado" no es transitiva, por lo que se cierra
transitivamente con una estructura **union-find**, obteniendo grupos
$\mathcal{G}_1,\dots,\mathcal{G}_m$. Para evitar la comparación $O(n^2)$ se usa
un esquema LSH: la huella se parte en cuatro bloques de 16 bits y solo se
comparan huellas que coincidan exactamente en algún bloque; por el principio
del palomar, dos huellas a distancia $\le 6$ comparten al menos un bloque
cuando el umbral se reparte entre los cuatro.

**Resultado medido:** $1{.}554$ grupos para $2{.}500$ imágenes de Kaggle, es
decir una **tasa de duplicación aparente del 37,8 %**. Con partición aleatoria,
la contaminación esperada del conjunto de prueba es sustancial.

### 8.3 Partición y validación cruzada por grupos

La restricción es $\mathcal{G}_i \subseteq \text{train} \;\vee\;
\mathcal{G}_i \subseteq \text{val} \;\vee\; \mathcal{G}_i \subseteq \text{test}$.
Se satisface con `StratifiedGroupKFold`, que además mantiene aproximadamente
constante la proporción de clases, estratificando por
fruta $\times$ calidad $\times$ fuente. La misma restricción se aplica en los
5 pliegues internos de `GridSearchCV`, porque de nada sirve una partición
limpia si la selección de hiperparámetros se hace con pliegues contaminados.

### 8.4 Protocolo de tres conjuntos

$$
\underbrace{70\,\%}_{\text{entrenar } \hat{h} \text{ y ajustar hiperparámetros por CV}} \quad
\underbrace{15\,\%}_{\text{comparar modelos entre sí}} \quad
\underbrace{15\,\%}_{\text{estimación final, se mira UNA vez}}
$$

Si se eligiera el modelo mirando el conjunto de prueba, su métrica sería
optimista por **selección múltiple**: al comparar $m$ modelos, el máximo de $m$
estimaciones ruidosas está sesgado hacia arriba.

*Código: `src/features/build_features.py`*

---

## 9. Resumen de notación

| Símbolo | Significado |
|---|---|
| $I$ | Imagen de entrada, $H \times W \times 3$ |
| $M$ | Máscara binaria de la fruta |
| $\mathcal{M}$, $N_\mathcal{M}$ | Conjunto de píxeles de la fruta y su cardinal |
| $\sigma, \varphi, g$ | Segmentación, extracción de características, clasificador |
| $\varphi(I) \in \mathbb{R}^{93}$ | Vector de características |
| $A, P$ | Área y perímetro del contorno principal |
| $d_{\text{px}}, d_{\text{rel}}, D$ | Diámetro en píxeles, normalizado y físico |
| $K$, $N_k$, $\alpha_k$ | Número de clases, soporte y peso de la clase $k$ |
| $C_{ij}$ | Costo de predecir $j$ cuando la verdad es $i$ |
| $\mathcal{G}_i$ | Grupo de imágenes casi duplicadas |
| $d_H$ | Distancia de Hamming |
