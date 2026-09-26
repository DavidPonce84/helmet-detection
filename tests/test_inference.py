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

    def test_legal_rag_coip_infraction(self):
        """Verifica que el RAG legal cite el Art. 389 num. 11 del COIP y las sanciones de tránsito."""
        from src.legal_rag import generate_legal_verdict
        res = generate_legal_verdict("DENEGADO", "Sin casco", 92.5)
        self.assertIn("389", res["articulo"])
        self.assertIn("30%", res["sancion_multa"])
        self.assertIn("-6 puntos", res["sancion_puntos"])
        self.assertIn("DICTAMEN SANCIONATORIO", res["dictamen"])

    def test_legal_rag_compliance(self):
        """Verifica que el RAG legal emita dictamen de conformidad cuando tiene casco."""
        from src.legal_rag import generate_legal_verdict
        res = generate_legal_verdict("PERMITIDO", "Con casco", 96.0)
        self.assertIn("DICTAMEN DE CONFORMIDAD", res["dictamen"])
        self.assertIn("$0", res["sancion_multa"])


if __name__ == "__main__":
    unittest.main()
