import cv2
import numpy as np

def resize_and_normalize(image_path, target_size=(128, 128)):
    """
    Carga una imagen desde una ruta en el disco, la redimensiona y normaliza 
    sus píxeles al rango flotante [0, 1].
    
    Parameters:
        image_path (str): Ruta local de la imagen.
        target_size (tuple): Dimensión fija (ancho, alto). Por defecto (128, 128) computacionalmente es mas facil por lo que es 2 a la 7.
        
    Returns:
        np.ndarray: Matriz de la imagen procesada en espacio RGB de forma (128, 128, 3) 
                    con valores flotantes entre 0.0 y 1.0.
    """
    # 1. Cargar la imagen usando OpenCV (se carga por defecto en formato BGR, habria sido directamente a rgb si hubieramos usado pil en vez de opencv)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"No se pudo cargar la imagen en la ruta: {image_path}")
        
    # 2. Convertir el espacio de color de BGR a RGB (formato estándar para matplotlib y redes neuronales)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 3. Redimensionar (Resize) la imagen a la dimensión objetivo deseada
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    
    # 4. Normalización: Escalar la intensidad de píxeles de [0, 255] a [0.0, 1.0]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return img_normalized

def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    Extrae un histograma de color 3D en el espacio RGB y lo normaliza.
    Devuelve un vector numérico de características de color.
    """
    # Denormalizar temporalmente si la imagen viene en rango [0, 1]
    if image.max() <= 1.0:
        img_uint8 = (image * 255).astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)
        
    # Calcular el histograma 3D para los 3 canales de color
    hist = cv2.calcHist([img_uint8], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    
    # Normalizar el histograma para que la suma de frecuencias sea 1
    cv2.normalize(hist, hist)
    
    # Aplanar la matriz 3D a un vector 1D
    return hist.flatten()

def extract_combined_features(image_path, target_size=(128, 128)):
    """
    Pipeline completo de preprocesamiento y extracción de características
    para una sola imagen.
    """
    # 1. Preprocesar la imagen (resize + normalizar)
    img_norm = resize_and_normalize(image_path, target_size)
    
    # 2. Extraer el histograma de color
    color_features = extract_color_histogram(img_norm)
    
    return color_features


def build_feature_dataset(df):
    """
    Recorre el DataFrame de imágenes, extrae vectores de 512 características
    y los consolida en matrices NumPy para Machine Learning.
    """
    X = []
    y = df['quality'].values  # O la etiqueta de calidad deseada
    
    for path in df['path']:
        features = extract_combined_features(path)
        X.append(features)
        
    return np.array(X), np.array(y)