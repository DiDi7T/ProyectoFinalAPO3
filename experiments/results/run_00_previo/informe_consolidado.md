# Informe consolidado de resultados

Generado automaticamente el 2026-11-26 01:30 por `python -m src.evaluation.evaluate`.

## Conjunto de datos

- Imagenes totales: **3362**
- Caracteristicas por imagen: **93**
- Grupos de imagenes casi duplicadas: **2124**
- Particiones: {'train': 2224, 'test': 673, 'val': 465}
- Clases de calidad: {'good': 1652, 'bad': 1483, 'regular': 227}
- Clases de tamano: {'mediano': 1441, 'grande': 1023, 'pequeño': 898}

### Composicion por fuente y calidad

| source      |   bad |   good |   regular |   TOTAL |
|:------------|------:|-------:|----------:|--------:|
| kaggle      |  1250 |   1250 |         0 |    2500 |
| recolectado |   233 |    402 |       227 |     862 |
| TOTAL       |  1483 |   1652 |       227 |    3362 |

### Fuente por particion

| split   |   kaggle |   recolectado |   TOTAL |
|:--------|---------:|--------------:|--------:|
| test    |      452 |           221 |     673 |
| train   |     1694 |           530 |    2224 |
| val     |      354 |           111 |     465 |
| TOTAL   |     2500 |           862 |    3362 |

## Tarea: calidad

### Comparacion de todos los modelos en el conjunto de prueba

| familia       | modelo          |   accuracy |   balanced_accuracy |   f1_macro |   cohen_kappa |    mcc |   roc_auc_ovr |   cv_f1_macro |   cv_f1_macro_train |   falsos_buenos |   brecha_train_test |
|:--------------|:----------------|-----------:|--------------------:|-----------:|--------------:|-------:|--------------:|--------------:|--------------------:|----------------:|--------------------:|
| ML clasico    | svm_rbf         |     0.9525 |              0.9037 |     0.9268 |        0.9132 | 0.9137 |        0.9882 |        0.9217 |              0.9938 |               5 |              0.067  |
| ML clasico    | knn             |     0.9331 |              0.8909 |     0.8946 |        0.879  | 0.8801 |        0.969  |        0.9097 |              0.9987 |               5 |              0.1041 |
| deep learning | cnn             |     0.9153 |              0.8849 |     0.8572 |        0.8488 | 0.8498 |        0.9783 |        0.8565 |              0.735  |               8 |             -0.1222 |
| ML clasico    | hgb             |     0.9302 |              0.8305 |     0.8526 |        0.8719 | 0.8722 |        0.9797 |        0.9145 |              0.9987 |               8 |              0.1461 |
| ML clasico    | rf              |     0.9079 |              0.7845 |     0.8157 |        0.8292 | 0.8316 |        0.9801 |        0.9048 |              0.9978 |              24 |              0.1821 |
| ML clasico    | logreg          |     0.8574 |              0.8159 |     0.7684 |        0.7498 | 0.7528 |        0.9529 |        0.7955 |              0.8729 |              14 |              0.1045 |
| deep learning | cnn (sin fondo) |     0.844  |              0.7699 |     0.7407 |        0.7233 | 0.7244 |        0.9245 |      nan      |            nan      |              24 |            nan      |
| linea base    | dummy           |     0.5141 |              0.3333 |     0.2264 |        0      | 0      |        0.5    |        0.2168 |              0.2168 |             286 |             -0.0095 |

### Generalizacion entre dominios (mejor modelo clasico)

| fuente      |   n |   accuracy |   balanced_accuracy |   f1_macro | clases_presentes   |
|:------------|----:|-----------:|--------------------:|-----------:|:-------------------|
| kaggle      | 452 |     0.9558 |              0.9556 |     0.6398 | good, regular, bad |
| recolectado | 221 |     0.8778 |              0.8309 |     0.8312 | good, regular, bad |
| TODAS       | 673 |     0.9302 |              0.8305 |     0.8526 | good, regular, bad |

La diferencia entre las filas `kaggle` y `recolectado` mide el cambio de dominio: las imagenes publicas estan tomadas en condiciones controladas, mientras que las propias tienen fondos, sombras e iluminacion reales.

### Desempeno por fruta (mejor modelo clasico)

| fruta   |   n |   accuracy |   f1_macro |   errores |
|:--------|----:|-----------:|-----------:|----------:|
| guava   | 114 |     0.8684 |     0.7474 |        15 |
| orange  | 162 |     0.8889 |     0.7976 |        18 |
| apple   | 163 |     0.9202 |     0.8321 |        13 |
| banana  |  89 |     0.9888 |     0.9737 |         1 |
| lime    | 145 |     1      |     1      |         0 |

## Tarea: tamano

### Comparacion de todos los modelos en el conjunto de prueba

| familia       | modelo   |   accuracy |   balanced_accuracy |   f1_macro |   cohen_kappa |    mcc |   roc_auc_ovr |   mae_ordinal |   cv_f1_macro |   cv_f1_macro_train | falsos_buenos   |   brecha_train_test |
|:--------------|:---------|-----------:|--------------------:|-----------:|--------------:|-------:|--------------:|--------------:|--------------:|--------------------:|:----------------|--------------------:|
| ML clasico    | rf       |     0.9985 |              0.9985 |     0.9988 |        0.9976 | 0.9976 |        1      |        0.0015 |        0.9991 |              1      |                 |              0.0012 |
| ML clasico    | hgb      |     0.9985 |              0.9985 |     0.9988 |        0.9976 | 0.9976 |        1      |        0.0015 |        0.9977 |              1      |                 |              0.0012 |
| ML clasico    | logreg   |     0.9465 |              0.95   |     0.9495 |        0.9129 | 0.9134 |        0.9864 |        0.0535 |        0.9627 |              0.9954 |                 |              0.046  |
| ML clasico    | svm_rbf  |     0.8529 |              0.849  |     0.8512 |        0.7599 | 0.7605 |        0.9707 |        0.1471 |        0.886  |              0.9992 |                 |              0.148  |
| deep learning | cnn      |     0.6969 |              0.7065 |     0.6877 |        0.5219 | 0.5309 |        0.8963 |        0.3091 |      nan      |            nan      |                 |            nan      |
| ML clasico    | knn      |     0.6776 |              0.6862 |     0.669  |        0.4875 | 0.4925 |        0.8138 |        0.3328 |        0.788  |              1      |                 |              0.331  |
| linea base    | dummy    |     0.4933 |              0.3333 |     0.2202 |        0      | 0      |        0.5    |        0.5067 |        0.1961 |              0.1961 |                 |             -0.0241 |

### Generalizacion entre dominios (mejor modelo clasico)

| fuente      |   n |   accuracy |   balanced_accuracy |   f1_macro | clases_presentes         |
|:------------|----:|-----------:|--------------------:|-----------:|:-------------------------|
| kaggle      | 452 |     1      |              1      |     1      | pequeño, mediano, grande |
| recolectado | 221 |     0.9955 |              0.998  |     0.9958 | pequeño, mediano, grande |
| TODAS       | 673 |     0.9985 |              0.9985 |     0.9988 | pequeño, mediano, grande |

La diferencia entre las filas `kaggle` y `recolectado` mide el cambio de dominio: las imagenes publicas estan tomadas en condiciones controladas, mientras que las propias tienen fondos, sombras e iluminacion reales.

### Desempeno por fruta (mejor modelo clasico)

| fruta   |   n |   accuracy |   f1_macro |   errores |
|:--------|----:|-----------:|-----------:|----------:|
| apple   | 163 |     0.9939 |     0.9947 |         1 |
| banana  |  89 |     1      |     1      |         0 |
| guava   | 114 |     1      |     1      |         0 |
| lime    | 145 |     1      |     1      |         0 |
| orange  | 162 |     1      |     1      |         0 |

## Como leer estos numeros

- **f1_macro** es la metrica de seleccion: promedia el F1 de cada clase sin ponderar por soporte, asi que una clase minoritaria pesa igual que una mayoritaria.
- **brecha_train_test** es la diferencia entre el F1 de entrenamiento y el de prueba. Valores por encima de ~0,05 indican sobreajuste.
- **falsos_buenos** cuenta las frutas realmente malas clasificadas como buenas. Es el error costoso del negocio y conviene minimizarlo incluso sacrificando algo de exactitud global.
- Las particiones respetan grupos de imagenes casi duplicadas, por lo que estas cifras no estan infladas por fuga de informacion.
