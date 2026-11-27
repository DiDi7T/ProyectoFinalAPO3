# Informe consolidado de resultados

Generado automaticamente el 2026-11-26 21:09 por `python -m src.evaluation.evaluate`.

## Conjunto de datos

- Imagenes totales: **3501**
- Caracteristicas por imagen: **93**
- Grupos de imagenes casi duplicadas: **2123**
- Particiones: {'train': 2318, 'val': 720, 'test': 463}
- Clases de calidad: {'good': 1672, 'bad': 1483, 'regular': 346}
- Clases de tamano: {'mediano': 1499, 'pequeño': 1003, 'grande': 999}

### Composicion por fuente y calidad

| source      |   bad |   good |   regular |   TOTAL |
|:------------|------:|-------:|----------:|--------:|
| kaggle      |  1250 |   1250 |         0 |    2500 |
| recolectado |   233 |    422 |       346 |    1001 |
| TOTAL       |  1483 |   1672 |       346 |    3501 |

### Fuente por particion

| split   |   kaggle |   recolectado |   TOTAL |
|:--------|---------:|--------------:|--------:|
| test    |      341 |           122 |     463 |
| train   |     1674 |           644 |    2318 |
| val     |      485 |           235 |     720 |
| TOTAL   |     2500 |          1001 |    3501 |

## Tarea: calidad

### Comparacion de todos los modelos en el conjunto de prueba

| familia       | modelo          |   accuracy |   balanced_accuracy |   f1_macro |   cohen_kappa |    mcc |   roc_auc_ovr |   cv_f1_macro |   cv_f1_macro_train |   falsos_buenos |   brecha_train_test |
|:--------------|:----------------|-----------:|--------------------:|-----------:|--------------:|-------:|--------------:|--------------:|--------------------:|----------------:|--------------------:|
| ML clasico    | hgb             |     0.9309 |              0.902  |     0.9119 |        0.8804 | 0.8805 |        0.9836 |        0.8886 |              0.9992 |              12 |              0.0873 |
| ML clasico    | knn             |     0.9114 |              0.8943 |     0.8913 |        0.848  | 0.8484 |        0.9644 |        0.8778 |              0.9992 |              11 |              0.1079 |
| ML clasico    | svm_rbf         |     0.9114 |              0.8875 |     0.8854 |        0.8477 | 0.8477 |        0.9814 |        0.8823 |              0.9858 |              13 |              0.1004 |
| ML clasico    | rf              |     0.9028 |              0.8636 |     0.8663 |        0.8322 | 0.8325 |        0.9783 |        0.8667 |              0.9949 |              16 |              0.1286 |
| deep learning | cnn             |     0.8683 |              0.9037 |     0.8304 |        0.7845 | 0.7922 |        0.9768 |        0.8914 |              0.8652 |              13 |              0.0348 |
| ML clasico    | logreg          |     0.8251 |              0.8373 |     0.7844 |        0.711  | 0.7157 |        0.9409 |        0.7864 |              0.8493 |              17 |              0.0649 |
| deep learning | cnn (sin fondo) |     0.6695 |              0.7251 |     0.628  |        0.4976 | 0.528  |        0.8814 |      nan      |            nan      |              11 |            nan      |
| linea base    | dummy           |     0.4838 |              0.3333 |     0.2174 |        0      | 0      |        0.5    |        0.2139 |              0.2139 |             194 |             -0.0035 |

### Generalizacion entre dominios (mejor modelo clasico)

| fuente      |   n |   accuracy |   balanced_accuracy |   f1_macro | clases_presentes   |
|:------------|----:|-----------:|--------------------:|-----------:|:-------------------|
| kaggle      | 341 |     0.9501 |              0.9494 |     0.9499 | good, bad          |
| recolectado | 122 |     0.877  |              0.8803 |     0.8762 | good, regular, bad |
| TODAS       | 463 |     0.9309 |              0.902  |     0.9119 | good, regular, bad |

La diferencia entre las filas `kaggle` y `recolectado` mide el cambio de dominio: las imagenes publicas estan tomadas en condiciones controladas, mientras que las propias tienen fondos, sombras e iluminacion reales.

### Desempeno por fruta (mejor modelo clasico)

| fruta   |   n |   accuracy |   f1_macro |   errores |
|:--------|----:|-----------:|-----------:|----------:|
| lime    |  91 |     0.9121 |     0.6251 |         8 |
| orange  |  92 |     0.9239 |     0.8984 |         7 |
| apple   |  99 |     0.9192 |     0.9272 |         8 |
| guava   |  93 |     0.9462 |     0.9388 |         5 |
| banana  |  88 |     0.9545 |     0.9654 |         4 |

## Tarea: tamano

### Comparacion de todos los modelos en el conjunto de prueba

| familia       | modelo   |   accuracy |   balanced_accuracy |   f1_macro |   cohen_kappa |    mcc |   roc_auc_ovr |   mae_ordinal |   cv_f1_macro |   cv_f1_macro_train | falsos_buenos   |   brecha_train_test |
|:--------------|:---------|-----------:|--------------------:|-----------:|--------------:|-------:|--------------:|--------------:|--------------:|--------------------:|:----------------|--------------------:|
| ML clasico    | rf       |     0.9503 |              0.9583 |     0.952  |        0.9244 | 0.9265 |        0.9867 |        0.0518 |        0.9545 |              1      |                 |              0.048  |
| ML clasico    | hgb      |     0.946  |              0.9549 |     0.9476 |        0.9179 | 0.9201 |        0.9895 |        0.0562 |        0.9616 |              1      |                 |              0.0524 |
| ML clasico    | logreg   |     0.8812 |              0.8961 |     0.8851 |        0.8203 | 0.8245 |        0.9581 |        0.121  |        0.8906 |              0.957  |                 |              0.0719 |
| ML clasico    | svm_rbf  |     0.8013 |              0.8143 |     0.8064 |        0.6981 | 0.6994 |        0.9436 |        0.2009 |        0.8595 |              0.9812 |                 |              0.1748 |
| deep learning | cnn      |     0.7689 |              0.7781 |     0.7757 |        0.647  | 0.6476 |        0.922  |        0.2333 |      nan      |            nan      |                 |            nan      |
| ML clasico    | knn      |     0.7451 |              0.7496 |     0.7487 |        0.6086 | 0.6099 |        0.8557 |        0.2592 |        0.7582 |              1      |                 |              0.2513 |
| linea base    | dummy    |     0.432  |              0.3333 |     0.2011 |        0      | 0      |        0.5    |        0.568  |        0.1948 |              0.1948 |                 |             -0.0063 |

### Generalizacion entre dominios (mejor modelo clasico)

| fuente      |   n |   accuracy |   balanced_accuracy |   f1_macro | clases_presentes         |
|:------------|----:|-----------:|--------------------:|-----------:|:-------------------------|
| kaggle      | 341 |     0.9912 |              0.9935 |     0.9912 | pequeño, mediano, grande |
| recolectado | 122 |     0.8197 |              0.5288 |     0.5349 | pequeño, mediano, grande |
| TODAS       | 463 |     0.946  |              0.9549 |     0.9476 | pequeño, mediano, grande |

La diferencia entre las filas `kaggle` y `recolectado` mide el cambio de dominio: las imagenes publicas estan tomadas en condiciones controladas, mientras que las propias tienen fondos, sombras e iluminacion reales.

### Desempeno por fruta (mejor modelo clasico)

| fruta   |   n |   accuracy |   f1_macro |   errores |
|:--------|----:|-----------:|-----------:|----------:|
| lime    |  91 |     0.9451 |     0.6467 |         5 |
| apple   |  99 |     0.9192 |     0.8955 |         8 |
| banana  |  88 |     0.9318 |     0.9255 |         6 |
| orange  |  92 |     0.9348 |     0.9272 |         6 |
| guava   |  93 |     1      |     1      |         0 |

## Como leer estos numeros

- **f1_macro** es la metrica de seleccion: promedia el F1 de cada clase sin ponderar por soporte, asi que una clase minoritaria pesa igual que una mayoritaria.
- **brecha_train_test** es la diferencia entre el F1 de entrenamiento y el de prueba. Valores por encima de ~0,05 indican sobreajuste.
- **falsos_buenos** cuenta las frutas realmente malas clasificadas como buenas. Es el error costoso del negocio y conviene minimizarlo incluso sacrificando algo de exactitud global.
- Las particiones respetan grupos de imagenes casi duplicadas, por lo que estas cifras no estan infladas por fuga de informacion.
