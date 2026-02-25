"""
Router for Question 5: ¿Cuál ha sido el desempeño de las últimas tres campañas?

Cross-campaign performance analysis for the full portfolio.
Compatible with Python 3.9.20.
"""

import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .base_question_router import run_query_as_text, get_categorical_values

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/questions", tags=["question_5"])

SAMPLE_QUESTION = "¿Cuál ha sido el desempeño de las últimas tres campañas?"

SAMPLE_ANSWER = """El desempeño del portafolio completo en las últimas tres campañas muestra tendencias diferenciadas por producto, con un panorama general de redistribución de valor hacia productos de mayor rentabilidad, pero con desafíos operativos que limitan el potencial total.
Desempeño Consolidado del Portafolio:
Campaña C3-2025: Total elegibles: 12,252,558 | Asignadas: 11,372,040 | Eficiencia: 92.8% | NPV total: $31,847M
Campaña C4-2025: Total elegibles: 12,172,643 | Asignadas: 11,145,496 | Eficiencia: 91.6% | NPV total: $31,234M
Campaña C5-2025: Total elegibles: 12,294,461 | Asignadas: 11,260,860 | Eficiencia: 91.6% | NPV total: $31,589M
Hallazgos Clave: Redistribución de valor hacia productos de mayor NPV unitario (CNC, AAC). Deterioro en eficiencia overall de 1.2pp. Conflictos internos de reglas de priorización. Volumen total estancado. NPV total resiliente."""

SQL_OVERALL_BY_CAMPAIGN = """
SELECT
    campania,
    SUM(conteo) AS total_elegibles,
    SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) AS asignadas,
    SUM(CASE WHEN flag_declinado = 1 THEN conteo ELSE 0 END) AS no_asignadas,
    ROUND(100.0 * SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) / SUM(conteo), 2) AS pct_asignacion,
    ROUND(SUM(npv), 2) AS npv_total
FROM cubo
GROUP BY campania
ORDER BY campania DESC
LIMIT 3
"""

SQL_BY_PRODUCT_LAST3 = """
SELECT
    campania,
    producto,
    SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) AS asignadas,
    ROUND(100.0 * SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) / SUM(conteo), 2) AS pct_asignacion,
    ROUND(SUM(npv), 2) AS npv_total
FROM cubo
WHERE campania IN (SELECT DISTINCT campania FROM cubo ORDER BY campania DESC LIMIT 3)
GROUP BY campania, producto
ORDER BY producto, campania
"""

SQL_DECLINE_TRENDS = """
SELECT
    campania,
    causa_no_asignacion,
    SUM(conteo) AS total_no_asignadas
FROM cubo
WHERE flag_declinado = 1
    AND causa_no_asignacion IS NOT NULL
    AND campania IN (SELECT DISTINCT campania FROM cubo ORDER BY campania DESC LIMIT 3)
GROUP BY campania, causa_no_asignacion
ORDER BY total_no_asignadas DESC
LIMIT 20
"""


class Question5Request(BaseModel):
    pass  # No parameters needed


@router.get("/q5/params")
async def question_5_params():
    return JSONResponse(content={})


@router.post("/q5/ask")
async def question_5_ask(req: Question5Request):
    overall = run_query_as_text(SQL_OVERALL_BY_CAMPAIGN)
    by_product = run_query_as_text(SQL_BY_PRODUCT_LAST3)
    decline_trends = run_query_as_text(SQL_DECLINE_TRENDS)

    data_context = f"""DATOS REALES - ÚLTIMAS 3 CAMPAÑAS:

RESUMEN OVERALL POR CAMPAÑA:
{overall}

DESGLOSE POR PRODUCTO Y CAMPAÑA:
{by_product}

TENDENCIAS DE NO ASIGNACIÓN:
{decline_trends}
"""

    system_context = f"""Eres un analista experto en productos de crédito y campañas NPV.
Responde en español con análisis de tendencias.

PREGUNTA DE REFERENCIA: {SAMPLE_QUESTION}

EJEMPLO DE RESPUESTA ESPERADA:
{SAMPLE_ANSWER}

Usa los DATOS REALES para generar una respuesta similar.
Incluye: desempeño consolidado por campaña, análisis por producto, hallazgos clave y conclusión.
"""

    from .llm_routes import model_manager, data_analyzer
    if not data_analyzer.is_loaded:
        from config import CHATBOT_CSV_PATH
        data_analyzer.load_data(CHATBOT_CSV_PATH)

    answer, inference_time = model_manager.generate_response(
        user_message="¿Cuál ha sido el desempeño de las últimas tres campañas?",
        data_context=system_context + "\n\n" + data_context,
    )

    return JSONResponse(content={
        "response": answer,
        "question_id": 5,
        "inference_time": inference_time,
        "data_context": data_context,
    })
