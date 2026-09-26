"""
Motor Legal RAG (Retrieval-Augmented Generation) para Control de Tránsito en Nayón.
Indexa el COIP (Código Orgánico Integral Penal de Ecuador) y el Reglamento LOTTTSV
para generar dictámenes jurídicos automatizados con fundamentación legal y sanciones.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
LEGAL_DB_PATH = BASE_DIR / "data" / "legal_base.json"


def load_legal_knowledge_base() -> List[Dict[str, Any]]:
    """Carga los artículos legales indexados en JSON."""
    if not os.path.exists(LEGAL_DB_PATH):
        return []
    with open(LEGAL_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_relevant_legal_articles(query_text: str, verdict: str) -> List[Dict[str, Any]]:
    """
    Módulo Retrieval: Recupera los artículos pertinentes según el veredicto o texto.
    """
    corpus = load_legal_knowledge_base()
    if not corpus:
        return []

    query_lower = f"{query_text} {verdict}".lower()
    ranked_articles = []

    for item in corpus:
        score = 0
        for kw in item.get("palabras_clave", []):
            if kw in query_lower:
                score += 2
        # Ponderación según veredicto
        if verdict == "DENEGADO" and item["id"] == "COIP-389-11":
            score += 5
        elif verdict == "PERMITIDO" and "LOTTTSV" in item["id"]:
            score += 3
        
        ranked_articles.append((score, item))

    # Ordenar por relevancia
    ranked_articles.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in ranked_articles if score > 0] or corpus[:1]


def generate_legal_verdict(
    verdict: str,
    class_name: str,
    confidence_percent: float,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Módulo Generator: Genera el dictamen jurídico a partir del contexto legal recuperado.
    Soporta LLMs ultraligeros y cuenta con un sintetizador jurídico nativo inmediato.
    """
    articles = retrieve_relevant_legal_articles(class_name, verdict)
    primary_article = articles[0] if articles else None

    # Caso 1: Infracción detectada (Sin Casco)
    if verdict == "DENEGADO" or "sin casco" in class_name.lower():
        norma = "COIP Artículo 389, numeral 11"
        tipo = "Contravención de tránsito de cuarta clase"
        multa = "30% del Salario Básico Unificado (aprox. $138 USD)"
        puntos = "-6 puntos en la licencia de conducir"
        
        dictamen = (
            f"DICTAMEN SANCIONATORIO: Se evidencia que el conductor circula sin casco homologado "
            f"con un nivel de certeza técnica del {confidence_percent}%. Esta conducta vulnera de forma "
            f"directa el {norma}, constituyendo una {tipo}. Corresponde la aplicación de una {multa} "
            f"y la reducción de {puntos}, procediendo a detener la marcha del vehículo en la garita Nayón."
        )
        resumen = f"Infracción {norma}: Multa 30% SBU ($138) y -6 puntos en licencia."

    # Caso 2: Acceso Permitido (Con Casco)
    elif verdict == "PERMITIDO" or "con casco" in class_name.lower():
        norma = "Cumplimiento COIP Art. 389 (numeral 11) y LOTTTSV Art. 272"
        tipo = "Conducción en regla / Sin infracción"
        multa = "$0 (Sin sanción económica)"
        puntos = "0 puntos deducidos"
        
        dictamen = (
            f"DICTAMEN DE CONFORMIDAD: El motociclista utiliza casco de seguridad homologado "
            f"(Certeza del modelo: {confidence_percent}%), dando pleno cumplimiento a las normas "
            f"de seguridad vial vigentes en el territorio nacional. No existe mérito de sanción; "
            f"se autoriza el acceso inmediato por el puesto de control Nayón."
        )
        resumen = "Conductor en regla: Cumple normativa COIP/LOTTTSV. Acceso autorizado."

    # Caso 3: No Concluyente
    else:
        norma = "Protocolo de verificación preventiva de seguridad vial"
        tipo = "Inspección preliminar no concluyente"
        multa = "Pendiente de verificación"
        puntos = "Pendiente"
        
        dictamen = (
            f"DICTAMEN PREVENTIVO: La certeza visual del sistema ({confidence_percent}%) no alcanza "
            f"el umbral reglamentario requerido. Por principio de seguridad vial, se requiere que el "
            f"operador solicite al motociclista enfocar adecuadamente su cabeza para una segunda toma "
            f"antes de autorizar o denegar el paso."
        )
        resumen = "Inspección no concluyente: Reintentar captura antes de emitir dictamen legal."

    # Si se configuró una API key para un LLM externo ultraligero (ej. OpenAI gpt-4o-mini), se puede refinar
    if api_key and api_key.startswith("sk-"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = (
                f"Actúa como asesor jurídico de tránsito en Ecuador. Con base en este artículo del COIP:\n"
                f"{json.dumps(primary_article, ensure_ascii=False)}\n\n"
                f"Genera un dictamen formal conciso de máximo 3 frases para un motociclista que fue "
                f"detectado con estado '{class_name}' ({confidence_percent}% de certeza). Incluye la sanción exacta."
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.2
            )
            llm_text = response.choices[0].message.content.strip()
            if llm_text:
                dictamen = f"[LLM gpt-4o-mini] {llm_text}"
        except Exception:
            pass  # Fallback suave al dictamen base

    return {
        "articulo": norma,
        "tipo_infraccion": tipo,
        "sancion_multa": multa,
        "sancion_puntos": puntos,
        "dictamen": dictamen,
        "resumen_ejecutivo": resumen,
        "fuente_oficial": primary_article["conducta"] if primary_article else ""
    }
