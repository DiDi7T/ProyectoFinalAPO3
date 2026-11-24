from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Rutas base relativas a la raíz del proyecto (ProyectoFinalAPO3)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_PROCESSED = PROJECT_ROOT / "data_processed"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Mapeo de normalización para nombres de carpetas
FRUIT_ALIASES = {
    "apple": "apple",
    "banana": "banana",
    "guava": "guava",
    "lime": "lime",
    "orange": "orange",
    "orage": "orange",  
}

QUALITY_ALIASES = {
    "good quality_fruits": "good",
    "good_quality_fruits": "good",
    "bad quality_fruits": "bad",
    "bad_quality_fruits": "bad",
    "regular_quality_fruits": "regular",
}


def normalize_fruit_name(folder_name: str) -> str | None:
    """Extrae y normaliza el nombre de la fruta si pertenece a las 5 clases objetivo."""
    clean = folder_name.lower().strip()
    for alias, standard in FRUIT_ALIASES.items():
        if alias in clean:
            return standard
    return None


def get_fruit_mask_and_size(image_bgr: np.ndarray) -> tuple[np.ndarray, float, str]:
    """
    Aísla la fruta central separándola de fondos neutros, sombras y manos,
    y calcula el diámetro relativo normalizado.
    """
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, sat, val = cv2.split(hsv)

    # Desenfoque leve para eliminar textura del mesón (ej. granito)
    sat_blur = cv2.GaussianBlur(sat, (7, 7), 0)

    # Umbral de Otsu sobre saturación con piso mínimo de 45 para ignorar grises/sombras/guantes
    otsu_thresh, _ = cv2.threshold(sat_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final_thresh = max(otsu_thresh, 45)
    _, mask = cv2.threshold(sat_blur, final_thresh, 255, cv2.THRESH_BINARY)

    # Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Evitar que el fondo tocando los bordes se tome como objeto
    margin = max(2, int(min(h, w) * 0.02))
    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, 0.5, "mediano"

    # Priorizar contornos grandes cercanos al centro de la imagen
    img_center = np.array([w / 2.0, h / 2.0])
    best_contour = None
    best_score = -1.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < (h * w * 0.01):  # Ignorar ruido menor al 1% de la imagen
            continue
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            dist = np.linalg.norm(np.array([cx, cy]) - img_center) / np.hypot(w, h)
            score = area * (1.0 - 0.7 * dist)
            if score > best_score:
                best_score = score
                best_contour = c

    if best_contour is None:
        best_contour = max(contours, key=cv2.contourArea)

    clean_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(clean_mask, [best_contour], -1, 255, thickness=cv2.FILLED)

    area = cv2.contourArea(best_contour)
    equiv_diameter = np.sqrt(4 * area / np.pi)
    relative_diameter = float(np.clip(equiv_diameter / max(h, w), 0.0, 1.0))

    if relative_diameter < 0.35:
        size_label = "pequeño"
    elif relative_diameter <= 0.58:
        size_label = "mediano"
    else:
        size_label = "grande"

    return clean_mask, round(relative_diameter, 4), size_label


def scan_source(base_dir: Path, source_name: str) -> list[dict]:
    """Recorre una fuente de datos (kaggle o recolectado) y extrae metadatos válidos."""
    records = []
    if not base_dir.exists():
        print(f"[AVISO] No se encontró el directorio: {base_dir}")
        return records

    for quality_dir in sorted(base_dir.iterdir()):
        if not quality_dir.is_dir():
            continue

        quality_key = quality_dir.name.lower().strip()
        if quality_key not in QUALITY_ALIASES:
            # Ignora explícitamente 'Mixed Qualit_Fruits' u otras carpetas no mapeadas
            print(f"[INFO] Ignorando carpeta de calidad: {quality_dir.name}")
            continue

        quality_label = QUALITY_ALIASES[quality_key]

        for fruit_dir in sorted(quality_dir.iterdir()):
            if not fruit_dir.is_dir():
                continue

            fruit_label = normalize_fruit_name(fruit_dir.name)
            if fruit_label is None:
                # Ignora frutas fuera de las 5 clases (ej. pomegranate en Kaggle)
                print(f"[INFO] Ignorando fruta fuera de alcance: {fruit_dir.name}")
                continue

            for img_path in sorted(fruit_dir.rglob("*")):
                if img_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"[AVISO] Imagen corrupta o ilegible: {img_path}")
                    continue

                _, rel_diam, size_label = get_fruit_mask_and_size(img)
                rel_path = img_path.relative_to(PROJECT_ROOT).as_posix()

                records.append(
                    {
                        "filepath": rel_path,
                        "source": source_name,
                        "fruit": fruit_label,
                        "quality": quality_label,
                        "size_label": size_label,
                        "relative_diameter": rel_diam,
                    }
                )

    return records


def build_dataset() -> pd.DataFrame:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    kaggle_dir = DATA_RAW / "kaggle" / "archive"
    recolectado_dir = DATA_RAW / "recolectado"

    print("Escaneando dataset de Kaggle...")
    kaggle_records = scan_source(kaggle_dir, "kaggle")

    print("Escaneando dataset recolectado...")
    recolectado_records = scan_source(recolectado_dir, "recolectado")

    df = pd.DataFrame(kaggle_records + recolectado_records)
    output_csv = DATA_PROCESSED / "labels.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\nDataset consolidado guardado en: {output_csv}")
    print(f"Total de imágenes válidas: {len(df)}")
    if not df.empty:
        print("\nDistribución por Calidad y Fuente:")
        print(pd.crosstab(df["quality"], df["source"], margins=True))
        print("\nDistribución por Fruta y Calidad:")
        print(pd.crosstab(df["fruit"], df["quality"], margins=True))

    return df


if __name__ == "__main__":
    build_dataset()