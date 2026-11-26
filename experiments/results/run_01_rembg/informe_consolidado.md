# Informe consolidado de resultados

Generado automaticamente el 2026-11-26 14:26 por `python -m src.evaluation.evaluate`.

## Conjunto de datos

- Imagenes totales: **3336**
- Caracteristicas por imagen: **93**
- Grupos de imagenes casi duplicadas: **1964**
- Particiones: {'train': 2200, 'val': 696, 'test': 440}
- Clases de calidad: {'good': 1652, 'bad': 1457, 'regular': 227}
- Clases de tamano: {'mediano': 1437, 'grande': 993, 'pequeño': 906}

### Composicion por fuente y calidad

| source      |   bad |   good |   regular |   TOTAL |
|:------------|------:|-------:|----------:|--------:|
| kaggle      |  1250 |   1250 |         0 |    2500 |
| recolectado |   207 |    402 |       227 |     836 |
| TOTAL       |  1457 |   1652 |       227 |    3336 |

### Fuente por particion

| split   |   kaggle |   recolectado |   TOTAL |
|:--------|---------:|--------------:|--------:|
| test    |      341 |            99 |     440 |
| train   |     1674 |           526 |    2200 |
| val     |      485 |           211 |     696 |
| TOTAL   |     2500 |           836 |    3336 |

## Tarea: calidad

### Comparacion de todos los modelos en el conjunto de prueba

| familia       | modelo          |   accuracy |   balanced_accuracy |   f1_macro |   cohen_kappa |    mcc |   roc_auc_ovr |   cv_f1_macro |   cv_f1_macro_train |   falsos_buenos |   brecha_train_test |
|:--------------|:----------------|-----------:|--------------------:|-----------:|--------------:|-------:|--------------:|--------------:|--------------------:|----------------:|--------------------:|
| ML clasico    | svm_rbf         |     0.95   |              0.9639 |     0.9458 |        0.9107 | 0.9109 |        0.9868 |        0.8692 |              0.9944 |              11 |              0.0486 |
| ML clasico    | knn             |     0.9205 |              0.9238 |     0.9144 |        0.8578 | 0.8581 |        0.9884 |        0.8651 |              0.9988 |              13 |              0.0844 |
| ML clasico    | hgb             |     0.9455 |              0.8819 |     0.8974 |        0.9013 | 0.9016 |        0.9894 |        0.8834 |              0.9988 |               7 |              0.1015 |
| ML clasico    | rf              |     0.9114 |              0.9158 |     0.8786 |        0.843  | 0.8443 |        0.9863 |        0.8534 |              0.9778 |              17 |              0.0991 |
| deep learning | cnn             |     0.8841 |              0.9182 |     0.824  |        0.8024 | 0.809  |        0.9808 |        0.8696 |              0.8513 |               6 |              0.0272 |
| ML clasico    | logreg          |     0.8409 |              0.879  |     0.7851 |        0.7297 | 0.7371 |        0.9658 |        0.7483 |              0.823  |              14 |              0.0379 |
| deep learning | cnn (sin fondo) |     0.85   |              0.8751 |     0.7781 |        0.7473 | 0.7563 |        0.9683 |      nan      |            nan      |              11 |            nan      |
| linea base    | dummy           |     0.5023 |              0.3333 |     0.2229 |        0      | 0      |        0.5    |        0.2192 |              0.2192 |             190 |             -0.0037 |

### Generalizacion entre dominios (mejor modelo clasico)

| fuente      |   n |   accuracy |   balanced_accuracy |   f1_macro | clases_presentes   |
|:------------|----:|-----------:|--------------------:|-----------:|:-------------------|
| kaggle      | 341 |     0.9531 |              0.9533 |     0.6373 | good, regular, bad |
| recolectado |  99 |     0.9394 |              0.9361 |     0.9385 | good, regular, bad |
| TODAS       | 440 |     0.95   |              0.9639 |     0.9458 | good, regular, bad |

La diferencia entre las filas `kaggle` y `recolectado` mide el cambio de dominio: las imagenes publicas estan tomadas en condiciones controladas, mientras que las propias tienen fondos, sombras e iluminacion reales.

### Desempeno por fruta (mejor modelo clasico)

| fruta   |   n |   accuracy |   f1_macro |   errores |
|:--------|----:|-----------:|-----------:|----------:|
| apple   |  89 |     0.8876 |     0.8456 |        10 |
| orange  |  84 |     0.9405 |     0.9303 |         5 |
| lime    |  88 |     0.9773 |     0.9649 |         2 |
| banana  |  86 |     0.9651 |     0.9743 |         3 |
| guava   |  93 |     0.9785 |     0.9845 |         2 |

## Tarea: tamano

### Comparacion de todos los modelos en el conjunto de prueba

| familia       | modelo   |   accuracy |   balanced_accuracy |   f1_macro |   cohen_kappa |    mcc |   roc_auc_ovr |   mae_ordinal |   cv_f1_macro |   cv_f1_macro_train | falsos_buenos   |   brecha_train_test |
|:--------------|:---------|-----------:|--------------------:|-----------:|--------------:|-------:|--------------:|--------------:|--------------:|--------------------:|:----------------|--------------------:|
| ML clasico    | rf       |     1      |              1      |     1      |        1      | 1      |        1      |        0      |        0.9978 |              1      |                 |              0      |
| ML clasico    | hgb      |     0.9977 |              0.9969 |     0.9976 |        0.9965 | 0.9965 |        1      |        0.0023 |        0.9955 |              1      |                 |              0.0024 |
| ML clasico    | logreg   |     0.9205 |              0.9143 |     0.9182 |        0.8757 | 0.876  |        0.9805 |        0.0795 |        0.9544 |              0.9954 |                 |              0.0772 |
| ML clasico    | svm_rbf  |     0.8523 |              0.8419 |     0.8482 |        0.7682 | 0.769  |        0.9698 |        0.1477 |        0.8898 |              0.9876 |                 |              0.1393 |
| deep learning | cnn      |     0.8205 |              0.8493 |     0.825  |        0.7294 | 0.7408 |        0.9486 |        0.1818 |      nan      |            nan      |                 |            nan      |
| ML clasico    | knn      |     0.7523 |              0.7459 |     0.7506 |        0.6115 | 0.6137 |        0.85   |        0.25   |        0.7584 |              1      |                 |              0.2494 |
| linea base    | dummy    |     0.4545 |              0.3333 |     0.2083 |        0      | 0      |        0.5    |        0.5455 |        0.196  |              0.196  |                 |             -0.0124 |

### Generalizacion entre dominios (mejor modelo clasico)

| fuente      |   n |   accuracy |   balanced_accuracy |   f1_macro | clases_presentes         |
|:------------|----:|-----------:|--------------------:|-----------:|:-------------------------|
| kaggle      | 341 |          1 |                   1 |          1 | pequeño, mediano, grande |
| recolectado |  99 |          1 |                   1 |          1 | pequeño, mediano, grande |
| TODAS       | 440 |          1 |                   1 |          1 | pequeño, mediano, grande |

La diferencia entre las filas `kaggle` y `recolectado` mide el cambio de dominio: las imagenes publicas estan tomadas en condiciones controladas, mientras que las propias tienen fondos, sombras e iluminacion reales.

### Desempeno por fruta (mejor modelo clasico)

| fruta   |   n |   accuracy |   f1_macro |   errores |
|:--------|----:|-----------:|-----------:|----------:|
| apple   |  89 |          1 |          1 |         0 |
| banana  |  86 |          1 |          1 |         0 |
| guava   |  93 |          1 |          1 |         0 |
| lime    |  88 |          1 |          1 |         0 |
| orange  |  84 |          1 |          1 |         0 |

## Como leer estos numeros

- **f1_macro** es la metrica de seleccion: promedia el F1 de cada clase sin ponderar por soporte, asi que una clase minoritaria pesa igual que una mayoritaria.
- **brecha_train_test** es la diferencia entre el F1 de entrenamiento y el de prueba. Valores por encima de ~0,05 indican sobreajuste.
- **falsos_buenos** cuenta las frutas realmente malas clasificadas como buenas. Es el error costoso del negocio y conviene minimizarlo incluso sacrificando algo de exactitud global.
- Las particiones respetan grupos de imagenes casi duplicadas, por lo que estas cifras no estan infladas por fuga de informacion.
