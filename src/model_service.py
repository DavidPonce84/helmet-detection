"""
Servicio de inferencia y carga de modelo para control de cascos en Nayón.
Preprocesamiento según especificación de Teachable Machine (224x224, normalización [-1, 1]).
"""
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any
from PIL import Image, ImageOps
import numpy as np

# Rutas por defecto del modelo y etiquetas
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "model" / "keras_model.h5"
DEFAULT_LABELS_PATH = BASE_DIR / "model" / "labels.txt"


def load_labels(labels_path: Path = DEFAULT_LABELS_PATH) -> List[str]:
    """Carga y limpia las etiquetas del modelo."""
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Archivo de etiquetas no encontrado en {labels_path}")
    
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels


def load_prediction_model(model_path: Path = DEFAULT_MODEL_PATH):
    """
    Carga el modelo Keras exportado de Teachable Machine.
    Utiliza tf_keras para compatibilidad total con modelos H5 de Keras 2/Teachable Machine.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Archivo de modelo no encontrado en {model_path}")
    
    try:
        import tf_keras
        model = tf_keras.models.load_model(str(model_path), compile=False)
        return model
    except ImportError:
        from tensorflow.keras.models import load_model
        return load_model(str(model_path), compile=False)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesa una imagen PIL según el pipeline de Teachable Machine:
    1. Asegurar formato RGB.
    2. Recorte y ajuste al centro a 224x224.
    3. Normalización: (pixel / 127.5) - 1.0.
    4. Expansión a tensor de forma (1, 224, 224, 3).
    """
    # Asegurar modo RGB (por si viene en RGBA o escala de grises)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Recorte proporcional centrado
    target_size = (224, 224)
    image_fitted = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    
    # Convertir a numpy array float32
    img_array = np.asarray(image_fitted, dtype=np.float32)
    
    # Normalizar al rango [-1.0, 1.0]
    normalized_img = (img_array / 127.5) - 1.0
    
    # Expandir dimensiones para batch (1, 224, 224, 3)
    data = np.expand_dims(normalized_img, axis=0)
    return data


def predict_helmet(
    model, 
    labels: List[str], 
    image: Image.Image, 
    confidence_threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Ejecuta la inferencia sobre una imagen y retorna el dictamen de acceso.
    
    Retorna un diccionario con:
    - class_name: Nombre de la clase ganadora (ej. 'Con casco' o 'Sin casco')
    - raw_label: Texto original de la etiqueta (ej. '1 Con casco')
    - confidence: Certeza porcentual (0.0 a 1.0)
    - probabilities: Distribución de probabilidades de todas las clases
    - verdict: 'PERMITIDO', 'DENEGADO' o 'NO_CONCLUYENTE'
    - message: Mensaje explicativo para el operador
    - color: 'green', 'red' o 'amber'
    """
    tensor = preprocess_image(image)
    predictions = model.predict(tensor, verbose=0)[0]
    
    top_index = int(np.argmax(predictions))
    top_confidence = float(predictions[top_index])
    raw_label = labels[top_index] if top_index < len(labels) else f"Clase {top_index}"
    
    # Extraer nombre limpio (quitando número inicial ej. '0 Sin casco' -> 'Sin casco')
    parts = raw_label.split(" ", 1)
    clean_label = parts[1].strip() if len(parts) > 1 else raw_label
    
    # Evaluar reglas de negocio y umbral
    is_con_casco = "con casco" in clean_label.lower()
    
    if top_confidence < confidence_threshold:
        verdict = "NO_CONCLUYENTE"
        color = "amber"
        message = f"Inspección no concluyente: Confianza ({top_confidence*100:.1f}%) inferior al umbral mínimo ({confidence_threshold*100:.1f}%). Reintente la captura enfocando mejor la cabeza del conductor."
    elif is_con_casco:
        verdict = "PERMITIDO"
        color = "green"
        message = "ACCESO PERMITIDO - Motociclista con casco de seguridad verificado."
    else:
        verdict = "DENEGADO"
        color = "red"
        message = "ACCESO DENEGADO - Infracción detectada: Conductor sin casco."

    prob_dict = {}
    for i, prob in enumerate(predictions):
        lbl = labels[i] if i < len(labels) else f"Clase {i}"
        prob_dict[lbl] = float(prob)
        
    return {
        "class_name": clean_label,
        "raw_label": raw_label,
        "confidence": top_confidence,
        "confidence_percent": round(top_confidence * 100, 1),
        "probabilities": prob_dict,
        "verdict": verdict,
        "color": color,
        "message": message
    }
