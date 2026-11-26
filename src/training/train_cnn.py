"""
Entrenamiento de la CNN multitarea.

Protocolo
---------
* Las particiones son las mismas que usan los modelos clasicos (agrupadas por
  duplicados), de modo que la comparacion entre familias de modelos es justa.
* Optimizador AdamW con decaimiento de peso 1e-4 y planificador cosenoidal.
* Parada temprana sobre el F1 macro de calidad en validacion, con paciencia
  configurable. El criterio de parada NUNCA mira el conjunto de prueba.
* Al final se evalua una sola vez en prueba y se escriben metricas, curvas de
  entrenamiento, matrices de confusion y una galeria con los errores mas
  confiados para el analisis de errores.

Uso
---
    python -m src.training.train_cnn
    python -m src.training.train_cnn --epocas 40 --lr 1e-3
    python -m src.training.train_cnn --aplicar-mascara   # ablacion sin fondo
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.evaluation import metrics as M
from src.evaluation import plots as P
from src.models.cnn import (
    FrutasDataset,
    FruitQualityCNN,
    PerdidaMultitarea,
    pesos_de_clase,
    predecir,
)
from src.utils.config import (
    CHECKPOINTS,
    FEATURES_DIR,
    QUALITY_CLASSES,
    RANDOM_STATE,
    RESULTS,
    SIZE_CLASSES,
    ensure_dirs,
    set_global_seed,
)

FIGURAS = RESULTS / "figuras"


def cargar_tensores() -> dict:
    """Carga las imagenes cacheadas, las mascaras y las etiquetas."""
    ruta = FEATURES_DIR / "X_cnn_bgr_uint8.npy"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta: python -m src.features.build_features"
        )

    mascaras_path = FEATURES_DIR / "X_cnn_mask_uint8.npy"
    split = np.load(FEATURES_DIR / "split.npy", allow_pickle=True).astype(str)
    return {
        "imgs": np.load(ruta, mmap_mode="r"),
        "masks": np.load(mascaras_path, mmap_mode="r") if mascaras_path.exists() else None,
        "y_quality": np.load(FEATURES_DIR / "y_quality.npy"),
        "y_size": np.load(FEATURES_DIR / "y_size.npy"),
        "split": split,
    }


def _subconjunto(datos: dict, mascara_bool: np.ndarray) -> tuple:
    idx = np.where(mascara_bool)[0]
    imgs = np.asarray(datos["imgs"][idx])
    masks = np.asarray(datos["masks"][idx]) if datos["masks"] is not None else None
    return imgs, masks, datos["y_quality"][idx], datos["y_size"][idx], idx


def entrenar(
    epocas: int = 30,
    lote: int = 32,
    lr: float = 1e-3,
    decaimiento: float = 1e-4,
    dropout: float = 0.3,
    paciencia: int = 8,
    aplicar_mascara: bool = False,
    lambda_tamano: float = 0.5,
    dispositivo: str | None = None,
) -> dict:
    """Entrena la CNN y escribe metricas y figuras. Devuelve el informe."""
    ensure_dirs()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    set_global_seed()

    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    datos = cargar_tensores()

    tr = datos["split"] == "train"
    va = datos["split"] == "val"
    te = datos["split"] == "test"

    imgs_tr, masks_tr, yq_tr, ys_tr, _ = _subconjunto(datos, tr)
    imgs_va, masks_va, yq_va, ys_va, _ = _subconjunto(datos, va)
    imgs_te, masks_te, yq_te, ys_te, idx_te = _subconjunto(datos, te)

    ds_tr = FrutasDataset(
        imgs_tr, yq_tr, ys_tr, masks_tr, entrenamiento=True,
        aplicar_mascara=aplicar_mascara, semilla=RANDOM_STATE,
    )
    ds_va = FrutasDataset(imgs_va, yq_va, ys_va, masks_va, aplicar_mascara=aplicar_mascara)
    ds_te = FrutasDataset(imgs_te, yq_te, ys_te, masks_te, aplicar_mascara=aplicar_mascara)

    # num_workers=0 a proposito: en Windows cada worker reimporta el modulo y
    # con los arreglos ya en memoria no hay ganancia.
    dl_tr = DataLoader(ds_tr, batch_size=lote, shuffle=True, num_workers=0, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=64, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=64, shuffle=False, num_workers=0)

    modelo = FruitQualityCNN(
        n_clases_calidad=len(QUALITY_CLASSES),
        n_clases_tamano=len(SIZE_CLASSES),
        dropout=dropout,
    ).to(dispositivo)

    criterio = PerdidaMultitarea(
        pesos_calidad=pesos_de_clase(yq_tr, len(QUALITY_CLASSES)).to(dispositivo),
        pesos_tamano=pesos_de_clase(ys_tr, len(SIZE_CLASSES)).to(dispositivo),
        lambda_calidad=1.0,
        lambda_tamano=lambda_tamano,
    )
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=decaimiento)
    planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=epocas)

    print("=" * 70)
    print(f"CNN multitarea en {dispositivo}  |  parametros: {modelo.n_parametros():,}")
    print(f"train={len(ds_tr)}  val={len(ds_va)}  test={len(ds_te)}")
    print(f"aplicar_mascara={aplicar_mascara}  lambda_tamano={lambda_tamano}")
    print("=" * 70)

    historial = {
        "epoca": [], "loss_train": [], "loss_val": [],
        "f1_quality_train": [], "f1_quality_val": [], "f1_size_val": [], "lr": [],
    }
    mejor_f1 = -1.0
    mejor_epoca = -1
    sin_mejora = 0
    ruta_mejor = CHECKPOINTS / ("cnn_mejor_mascara.pt" if aplicar_mascara else "cnn_mejor.pt")

    for epoca in range(1, epocas + 1):
        modelo.train()
        t0 = time.perf_counter()
        suma_loss, n_lotes = 0.0, 0
        pred_tr, real_tr = [], []

        for x, y_calidad, y_tamano in dl_tr:
            x = x.to(dispositivo)
            y_calidad = y_calidad.to(dispositivo)
            y_tamano = y_tamano.to(dispositivo)

            optimizador.zero_grad(set_to_none=True)
            logits_q, logits_s = modelo(x)
            perdida, partes = criterio(logits_q, logits_s, y_calidad, y_tamano)
            perdida.backward()
            # Recorte de gradiente: evita pasos destructivos en los primeros
            # lotes, cuando BatchNorm aun no tiene estadisticas estables.
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=5.0)
            optimizador.step()

            suma_loss += partes["loss"]
            n_lotes += 1
            pred_tr.append(logits_q.argmax(1).detach().cpu().numpy())
            real_tr.append(y_calidad.cpu().numpy())

        planificador.step()

        f1_tr = f1_score(
            np.concatenate(real_tr), np.concatenate(pred_tr), average="macro", zero_division=0
        )

        # ---- validacion ----
        modelo.eval()
        suma_val, n_val = 0.0, 0
        with torch.no_grad():
            for x, y_calidad, y_tamano in dl_va:
                x = x.to(dispositivo)
                logits_q, logits_s = modelo(x)
                perdida_val, partes_val = criterio(
                    logits_q, logits_s, y_calidad.to(dispositivo), y_tamano.to(dispositivo)
                )
                suma_val += partes_val["loss"]
                n_val += 1

        salida_val = predecir(modelo, dl_va, dispositivo)
        f1_val = f1_score(
            salida_val["y_calidad"], salida_val["pred_calidad"], average="macro", zero_division=0
        )
        f1_val_size = f1_score(
            salida_val["y_tamano"], salida_val["pred_tamano"], average="macro", zero_division=0
        )

        historial["epoca"].append(epoca)
        historial["loss_train"].append(suma_loss / max(n_lotes, 1))
        historial["loss_val"].append(suma_val / max(n_val, 1))
        historial["f1_quality_train"].append(float(f1_tr))
        historial["f1_quality_val"].append(float(f1_val))
        historial["f1_size_val"].append(float(f1_val_size))
        historial["lr"].append(optimizador.param_groups[0]["lr"])

        marca = ""
        if f1_val > mejor_f1:
            mejor_f1, mejor_epoca, sin_mejora = float(f1_val), epoca, 0
            torch.save(
                {
                    "state_dict": modelo.state_dict(),
                    "epoca": epoca,
                    "f1_val_quality": mejor_f1,
                    "config": {
                        "dropout": dropout,
                        "aplicar_mascara": aplicar_mascara,
                        "n_clases_calidad": len(QUALITY_CLASSES),
                        "n_clases_tamano": len(SIZE_CLASSES),
                    },
                },
                ruta_mejor,
            )
            marca = "  <- mejor"
        else:
            sin_mejora += 1

        print(
            f"ep {epoca:3d}/{epocas}  loss {historial['loss_train'][-1]:.4f} / "
            f"{historial['loss_val'][-1]:.4f}  F1q {f1_tr:.4f} / {f1_val:.4f}  "
            f"F1t_val {f1_val_size:.4f}  {time.perf_counter() - t0:.1f}s{marca}"
        )

        if sin_mejora >= paciencia:
            print(f"Parada temprana en la epoca {epoca} (sin mejora durante {paciencia}).")
            break

    # ---------------- evaluacion final ----------------
    estado = torch.load(ruta_mejor, map_location=dispositivo, weights_only=False)
    modelo.load_state_dict(estado["state_dict"])
    print(f"\nMejor epoca: {mejor_epoca} (F1 macro calidad en validacion = {mejor_f1:.4f})")

    salida_val = predecir(modelo, dl_va, dispositivo)
    salida_te = predecir(modelo, dl_te, dispositivo)

    m_val_q = M.calcular_metricas(
        salida_val["y_calidad"], salida_val["pred_calidad"], QUALITY_CLASSES,
        salida_val["proba_calidad"],
    )
    m_test_q = M.calcular_metricas(
        salida_te["y_calidad"], salida_te["pred_calidad"], QUALITY_CLASSES,
        salida_te["proba_calidad"],
    )
    m_test_q["costo"] = M.costo_asimetrico(
        salida_te["y_calidad"], salida_te["pred_calidad"],
        idx_good=0, idx_bad=len(QUALITY_CLASSES) - 1,
    )
    m_test_s = M.calcular_metricas(
        salida_te["y_tamano"], salida_te["pred_tamano"], SIZE_CLASSES,
        salida_te["proba_tamano"], ordinal=True,
    )

    print("\n=== PRUEBA (calidad) ===")
    for k in ("accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa", "mcc", "roc_auc_ovr"):
        print(f"  {k:20s} {m_test_q.get(k)}")
    print("=== PRUEBA (tamano) ===")
    for k in ("accuracy", "balanced_accuracy", "f1_macro", "mae_ordinal"):
        print(f"  {k:20s} {m_test_s.get(k)}")

    sufijo = "_mascara" if aplicar_mascara else ""
    P.curvas_entrenamiento(
        historial, f"CNN multitarea{' (sin fondo)' if aplicar_mascara else ''}",
        FIGURAS / f"cnn_curvas{sufijo}",
    )
    P.matriz_confusion(
        m_test_q["matriz_confusion"], m_test_q["etiquetas_matriz"],
        "CNN - calidad (prueba)", FIGURAS / f"cm_quality_cnn{sufijo}",
    )
    P.matriz_confusion(
        m_test_s["matriz_confusion"], m_test_s["etiquetas_matriz"],
        "CNN - tamano (prueba)", FIGURAS / f"cm_size_cnn{sufijo}",
    )
    P.curvas_roc(
        salida_te["y_calidad"], salida_te["proba_calidad"], m_test_q["etiquetas_matriz"],
        "Curvas ROC - CNN (calidad, prueba)", FIGURAS / f"roc_quality_cnn{sufijo}",
    )

    # ---- galeria de errores: los fallos mas confiados ----
    errores = np.where(salida_te["pred_calidad"] != salida_te["y_calidad"])[0]
    if len(errores):
        confianza = salida_te["proba_calidad"].max(axis=1)[errores]
        peores = errores[np.argsort(confianza)[::-1][:10]]
        titulos = [
            f"real={QUALITY_CLASSES[salida_te['y_calidad'][i]]}\n"
            f"pred={QUALITY_CLASSES[salida_te['pred_calidad'][i]]} "
            f"({salida_te['proba_calidad'][i].max():.2f})"
            for i in peores
        ]
        P.galeria_errores([imgs_te[i] for i in peores], titulos, FIGURAS / f"errores_cnn{sufijo}")

    informe = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "modelo": "FruitQualityCNN",
        "dispositivo": dispositivo,
        "n_parametros": modelo.n_parametros(),
        "hiperparametros": {
            "epocas_max": epocas, "lote": lote, "lr": lr,
            "weight_decay": decaimiento, "dropout": dropout,
            "paciencia": paciencia, "lambda_tamano": lambda_tamano,
            "aplicar_mascara": aplicar_mascara,
        },
        "mejor_epoca": mejor_epoca,
        "n_train": int(len(ds_tr)), "n_val": int(len(ds_va)), "n_test": int(len(ds_te)),
        "historial": historial,
        "metricas_validacion_calidad": m_val_q,
        "metricas_prueba_calidad": m_test_q,
        "metricas_prueba_tamano": m_test_s,
        "indices_prueba_mal_clasificados": [int(i) for i in errores],
        "checkpoint": str(ruta_mejor),
    }
    destino = RESULTS / f"cnn_metrics{sufijo}.json"
    with open(destino, "w", encoding="utf-8") as fh:
        json.dump(informe, fh, indent=2, ensure_ascii=False)
    print(f"\nMetricas completas: {destino}")
    print(f"Checkpoint: {ruta_mejor}")
    return informe


def main() -> None:
    from src.utils.registro import registrar_ejecucion

    with registrar_ejecucion("train_cnn"):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--epocas", type=int, default=30)
        parser.add_argument("--lote", type=int, default=32)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--dropout", type=float, default=0.3)
        parser.add_argument("--paciencia", type=int, default=8)
        parser.add_argument("--lambda-tamano", type=float, default=0.5)
        parser.add_argument(
            "--aplicar-mascara",
            action="store_true",
            help="Ablacion: entrenar con la fruta aislada sobre fondo negro",
        )
        args = parser.parse_args()

        entrenar(
            epocas=args.epocas,
            lote=args.lote,
            lr=args.lr,
            dropout=args.dropout,
            paciencia=args.paciencia,
            aplicar_mascara=args.aplicar_mascara,
            lambda_tamano=args.lambda_tamano,
        )


if __name__ == "__main__":
    main()
