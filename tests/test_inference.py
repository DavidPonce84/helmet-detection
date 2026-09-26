"""
Pruebas de verificación de inferencia y robustez del modelo.
Rol: @QA-Tester
"""
import unittest
from PIL import Image
import numpy as np

from src.model_service import (
    load_prediction_model,
    load_labels,
    preprocess_image,
    predict_helmet,
)


class TestModelInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_prediction_model()
        cls.labels = load_labels()

    def test_labels_loaded(self):
        """Verifica que las etiquetas existan y contengan las dos clases esperadas."""
        self.assertGreaterEqual(len(self.labels), 2)
        has_sin = any("sin casco" in l.lower() for l in self.labels)
        has_con = any("con casco" in l.lower() for l in self.labels)
        self.assertTrue(has_sin, "Falta la etiqueta 'Sin casco'")
        self.assertTrue(has_con, "Falta la etiqueta 'Con casco'")

    def test_tensor_preprocessing_shape(self):
        """Verifica que cualquier imagen se normalice a (1, 224, 224, 3) con valores en [-1, 1]."""
        test_img = Image.new("RGB", (640, 480), color=(200, 100, 50))
        tensor = preprocess_image(test_img)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertGreaterEqual(float(np.min(tensor)), -1.0)
        self.assertLessEqual(float(np.max(tensor)), 1.0)

    def test_inference_structure(self):
        """Verifica la estructura y campos del dictamen de predicción."""
        test_img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        result = predict_helmet(self.model, self.labels, test_img, confidence_threshold=0.75)
        
        self.assertIn("verdict", result)
        self.assertIn("confidence_percent", result)
        self.assertIn("color", result)
        self.assertIn("probabilities", result)
        self.assertIn(result["verdict"], ["PERMITIDO", "DENEGADO", "NO_CONCLUYENTE"])

    def test_confidence_threshold_logic(self):
        """Verifica que si el umbral es 99.9%, el dictamen sea NO_CONCLUYENTE salvo certeza extrema."""
        test_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        result = predict_helmet(self.model, self.labels, test_img, confidence_threshold=0.9999)
        self.assertEqual(result["verdict"], "NO_CONCLUYENTE")
        self.assertEqual(result["color"], "amber")


if __name__ == "__main__":
    unittest.main()
