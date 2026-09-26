"""
Motor Legal RAG (Retrieval-Augmented Generation) para Control de Tránsito en Nayón.
Indexa el COIP (Código Orgánico Integral Penal de Ecuador) y el Reglamento LOTTTSV
para generar dictámenes jurídicos automatizados con fundamentación legal y sanciones.
Integra soporte para el LLM ligero de Command Code (con fallback inmediato de seguridad).
"""
import os
import json
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
LEGAL_DB_PATH = BASE_DIR / "data" / "legal_base.json"

COMMAND_CODE_API_KEY = "user_5YfhmMTP22utXunRWbZ8rBXzkiJmKvAPg7Z9NUaQ7VjBcxsz2Ku9seLw2wa4VwmkzAJBUYadUnXTMTWzfJ4xBwNP"
COMMAND_CODE_URL = "https://api.commandcode.ai/provider/v1/chat/completions"
COMMAND_CODE_MODEL = "poolside/laguna-s-2.1-free"


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


def query_command_code_llm(
    article: Dict[str, Any],
    class_name: str,
    verdict: str,
    confidence_percent: float,
    api_key: str = COMMAND_CODE_API_KEY
) -> Optional[str]:
    """
    Consulta al LLM ligero de Command Code para redactar el dictamen jurídico RAG.
    Usa timeout estricto de 4 segundos para garantizar que la UI nunca se congele.
    """
    if not api_key:
        return None

    prompt = (
        f"Eres el asesor legal de tránsito en la garita de Nayón, Ecuador. "
        f"Artículo aplicable: {article.get('articulo')} - {article.get('cuerpo_legal')}. "
        f"Conducta evaluada: {class_name} ({confidence_percent}% de certeza). "
        f"Sanción legal: {article.get('sancion_economica')} y {article.get('sancion_puntos')}. "
        f"Redacta un dictamen formal conciso de máximo 2 oraciones para el parte de tránsito."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Content-Type": "application/json"
    }

    payload = {
        "model": COMMAND_CODE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400
    }

    try:
        req = urllib.request.Request(
            COMMAND_CODE_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers
        )
        with urllib.request.urlopen(req, timeout=4) as res:
            data = json.loads(res.read().decode("utf-8"))
            msg = data.get("choices", [{}])[0].get("message", {})
            content = msg.get("content")
            if content and content.strip():
                return content.strip()
    except Exception:
        pass  # Fallback transparente y sin error

    return None


def generate_legal_verdict(
    verdict: str,
    class_name: str,
    confidence_percent: float,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Módulo Generator: Genera el dictamen jurídico a partir del contexto legal recuperado.
    Combina el LLM ligero de Command Code con un sintetizador legal nativo garantizado.
    """
    articles = retrieve_relevant_legal_articles(class_name, verdict)
    primary_article = articles[0] if articles else {}

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

    # Intentar enriquecer mediante el LLM de Command Code si la clave está activa
    effective_key = api_key or COMMAND_CODE_API_KEY
    if effective_key:
        llm_response = query_command_code_llm(
            article=primary_article,
            class_name=class_name,
            verdict=verdict,
            confidence_percent=confidence_percent,
            api_key=effective_key
        )
        if llm_response:
            dictamen = f"[Command Code AI] {llm_response}"

    return {
        "articulo": norma,
        "tipo_infraccion": tipo,
        "sancion_multa": multa,
        "sancion_puntos": puntos,
        "dictamen": dictamen,
        "resumen_ejecutivo": resumen,
        "fuente_oficial": primary_article.get("conducta", "")
    }
