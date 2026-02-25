"""
Router for Question 6: ¿Cuál es el porcentaje de asignación overall?

Overall assignment percentage with breakdown by product and decline analysis.
Compatible with Python 3.9.20.
"""

import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .base_question_router import run_query_as_text, get_categorical_values

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/questions", tags=["question_6"])

SAMPLE_QUESTION = "¿Cuál es el porcentaje de asignación overall?"

SAMPLE_ANSWER = """El porcentaje de asignación overall del portafolio completo en las últimas tres campañas:
C3-2025: 92.8% (Brecha: 880,518 ofertas no asignadas - 7.2%)
C4-2025: 91.6% (Brecha: 1,027,147 ofertas no asignadas - 8.4%)
C5-2025: 91.6% (Brecha: 1,033,601 ofertas no asignadas - 8.4%)
Se observa un deterioro de 1.2pp entre C3 y C4, con estabilización en C5.
Distribución de la brecha por producto (C5): AAC 42.2%, CPC 22.0%, TCC 20.1%, DC 8.8%, CNC 7.0%.
Comparativo de eficiencia: DC 95.3% (mejor), CNC 94.5%, CPC 92.0%, TCC 91.2%, AAC 88.7% (menor).
Oportunidad: llevar eficiencia de 91.6% a 93.5% capturaría ~230,000 ofertas adicionales."""

SQL_OVERALL_TREND = """
SELECT
    campania,
    SUM(conteo) AS total_elegibles,
    SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) AS asignadas,
    SUM(CASE WHEN flag_declinado = 1 THEN conteo ELSE 0 END) AS brecha,
    ROUND(100.0 * SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) / SUM(conteo), 2) AS pct_asignacion,
    ROUND(100.0 * SUM(CASE WHEN flag_declinado = 1 THEN conteo ELSE 0 END) / SUM(conteo), 2) AS pct_brecha
FROM cubo
GROUP BY campania
ORDER BY campania
"""

SQL_BRECHA_BY_PRODUCT = """
SELECT
    producto,
    SUM(CASE WHEN flag_declinado = 1 THEN conteo ELSE 0 END) AS no_asignadas,
    ROUND(100.0 * SUM(CASE WHEN flag_declinado = 1 THEN conteo ELSE 0 END) /
        (SELECT SUM(conteo) FROM cubo WHERE flag_declinado = 1 AND campania = c.campania), 2) AS pct_de_brecha_total,
    ROUND(100.0 * SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) / SUM(conteo), 2) AS eficiencia_producto
FROM cubo c
WHERE campania = ?
GROUP BY producto
ORDER BY no_asignadas DESC
"""

SQL_EFFICIENCY_RANKING = """
SELECT
    producto,
    ROUND(100.0 * SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) / SUM(conteo), 2) AS eficiencia
FROM cubo
GROUP BY producto
ORDER BY eficiencia DESC
"""


class Question6Request(BaseModel):
    campania: Optional[str] = None


@router.get("/q6/params")
async def question_6_params():
    return JSONResponse(content={
        "campania": get_categorical_values("campania"),
    })


@router.post("/q6/ask")
async def question_6_ask(req: Question6Request):
    campanias = get_categorical_values("campania")
    campania = req.campania or (campanias[-1] if campanias else "C5-2025")

    overall_trend = run_query_as_text(SQL_OVERALL_TREND)
    brecha = run_query_as_text(SQL_BRECHA_BY_PRODUCT, (campania,))
    ranking = run_query_as_text(SQL_EFFICIENCY_RANKING)

    data_context = f"""DATOS REALES:

TENDENCIA DE ASIGNACIÓN OVERALL POR CAMPAÑA:
{overall_trend}

DISTRIBUCIÓN DE LA BRECHA POR PRODUCTO (campaña {campania}):
{brecha}

RANKING DE EFICIENCIA POR PRODUCTO (TODAS LAS CAMPAÑAS):
{ranking}
"""

    system_context = f"""Eres un analista experto en productos de crédito y campañas NPV.
IMPORTANTE: SIEMPRE responde en español, sin importar en qué idioma se haga la pregunta.
Responde con análisis de eficiencia operativa.

PREGUNTA DE REFERENCIA: {SAMPLE_QUESTION}

EJEMPLO DE RESPUESTA ESPERADA:
{SAMPLE_ANSWER}

Usa los DATOS REALES para generar una respuesta similar.
Incluye: porcentaje overall por campaña, distribución de la brecha, ranking de eficiencia, implicaciones estratégicas.
"""

    from .llm_routes import model_manager, data_analyzer
    if not data_analyzer.is_loaded:
        from config import CHATBOT_CSV_PATH
        data_analyzer.load_data(CHATBOT_CSV_PATH)

    answer, inference_time = model_manager.generate_response(
        user_message=f"¿Cuál es el porcentaje de asignación overall? Enfócate en campaña {campania}.",
        data_context=system_context + "\n\n" + data_context,
    )

    return JSONResponse(content={
        "response": answer,
        "question_id": 6,
        "campania": campania,
        "inference_time": inference_time,
        "data_context": data_context,
    })
