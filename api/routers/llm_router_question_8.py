"""
Router for Question 8: ¿Cuál es la tendencia de las últimas N campañas?

Trend analysis across N campaigns with portfolio-level insights.
Compatible with Python 3.9.20.
"""

import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .base_question_router import run_query_as_text, get_categorical_values

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/questions", tags=["question_8"])

SAMPLE_QUESTION = "¿Cuál es la tendencia de las últimas N campañas?"

SAMPLE_ANSWER = """El portafolio está en un punto de inflexión crítico. Existe una redistribución natural hacia productos de mayor valor (CNC, AAC), pero esta migración está siendo frenada por límites operativos y conflictos de reglas de negocio. Los productos tradicionales (CPC, TCC) están en declive, pero este declive es en parte artificial (causado por problemas internos) y en parte estructural (cambios de mercado).
La tendencia actual es de estancamiento con redistribución. Sin intervenciones, el portafolio crecerá marginalmente (~1% anual) pero dejará más de $2,500 millones en NPV sobre la mesa. Con intervenciones focalizadas en los tres problemas principales (conflicto CPC-revolving, límites AAC, canibalización TCC), el portafolio podría crecer 7% anual y capturar $2,100 millones adicionales en NPV."""

SQL_CAMPAIGN_TREND = """
SELECT
    campania,
    SUM(conteo) AS total_elegibles,
    SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) AS asignadas,
    ROUND(100.0 * SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) / SUM(conteo), 2) AS pct_asignacion,
    ROUND(SUM(npv), 2) AS npv_total,
    ROUND(SUM(CASE WHEN flag_declinado = 0 THEN npv ELSE 0 END) / NULLIF(SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END), 0), 2) AS npv_promedio
FROM cubo
GROUP BY campania
ORDER BY campania DESC
LIMIT ?
"""

SQL_PRODUCT_TREND = """
SELECT
    producto,
    campania,
    SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) AS asignadas,
    ROUND(100.0 * SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) / SUM(conteo), 2) AS eficiencia,
    ROUND(SUM(npv), 2) AS npv_total
FROM cubo
WHERE campania IN (SELECT DISTINCT campania FROM cubo ORDER BY campania DESC LIMIT ?)
GROUP BY producto, campania
ORDER BY producto, campania
"""

SQL_DECLINE_TREND = """
SELECT
    campania,
    causa_no_asignacion,
    SUM(conteo) AS total
FROM cubo
WHERE flag_declinado = 1
    AND causa_no_asignacion IS NOT NULL
    AND campania IN (SELECT DISTINCT campania FROM cubo ORDER BY campania DESC LIMIT ?)
GROUP BY campania, causa_no_asignacion
ORDER BY campania, total DESC
"""


class Question8Request(BaseModel):
    n_campanias: Optional[int] = 3


@router.get("/q8/params")
async def question_8_params():
    campanias = get_categorical_values("campania")
    return JSONResponse(content={
        "n_campanias": list(range(2, len(campanias) + 1)),
    })


@router.post("/q8/ask")
async def question_8_ask(req: Question8Request):
    n = req.n_campanias or 3

    campaign_trend = run_query_as_text(SQL_CAMPAIGN_TREND, (n,))
    product_trend = run_query_as_text(SQL_PRODUCT_TREND, (n,))
    decline_trend = run_query_as_text(SQL_DECLINE_TREND, (n,))

    data_context = f"""DATOS REALES - ÚLTIMAS {n} CAMPAÑAS:

TENDENCIA OVERALL:
{campaign_trend}

TENDENCIA POR PRODUCTO:
{product_trend}

TENDENCIA DE CAUSAS DE NO ASIGNACIÓN:
{decline_trend}
"""

    system_context = f"""Eres un analista experto en productos de crédito y campañas NPV.
Responde en español con análisis de tendencias y proyecciones.

PREGUNTA DE REFERENCIA: {SAMPLE_QUESTION}

EJEMPLO DE RESPUESTA ESPERADA:
{SAMPLE_ANSWER}

Usa los DATOS REALES para generar una respuesta similar analizando las últimas {n} campañas.
Incluye: tendencia general, productos en crecimiento vs declive, causas principales, y conclusión con recomendaciones.
"""

    from .llm_routes import model_manager, data_analyzer
    if not data_analyzer.is_loaded:
        from config import CHATBOT_CSV_PATH
        data_analyzer.load_data(CHATBOT_CSV_PATH)

    answer, inference_time = model_manager.generate_response(
        user_message=f"¿Cuál es la tendencia de las últimas {n} campañas?",
        data_context=system_context + "\n\n" + data_context,
    )

    return JSONResponse(content={
        "response": answer,
        "question_id": 8,
        "n_campanias": n,
        "inference_time": inference_time,
        "data_context": data_context,
    })
