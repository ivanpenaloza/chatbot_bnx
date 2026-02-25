"""
Router for Question 2: ¿Qué porcentaje de las ofertas se asignó para [producto]?

Applies to all products. Shows assignment percentage with trend across campaigns.
Compatible with Python 3.9.20.
"""

import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .base_question_router import run_query_as_text, get_categorical_values

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/questions", tags=["question_2"])

SAMPLE_QUESTION = "¿Qué porcentaje de las ofertas se asignó para CPC? (Aplica para todos los productos)"

SAMPLE_ANSWER = """En la campaña C5-2025, el porcentaje de asignación para CPC fue del 92.0%.
Esto indica que de cada 100 clientes identificados como elegibles para recibir una oferta de CPC, se logró asignar la oferta a 92 de ellos. Los 8 restantes no recibieron la asignación por diversas razones.
Para contextualizar la evolución de este indicador:
C3-2025: 93.8% de asignación (mejor desempeño del período)
C4-2025: 91.5% de asignación (caída de 2.3 puntos porcentuales)
C5-2025: 92.0% de asignación (recuperación de 0.5 puntos porcentuales)
Si bien se ha mejorado respecto a C4, el desempeño continúa 1.8 puntos porcentuales por debajo del mejor resultado en C3. Aunque esta diferencia puede parecer marginal, al multiplicarla por el volumen de clientes representa decenas de miles de ofertas que no están siendo asignadas.
El 92% constituye una tasa razonablemente sólida, considerando que alcanzar el 100% no es realista debido a fricciones inherentes al proceso."""

SQL_ASSIGNMENT_BY_CAMPAIGN = """
SELECT
    campania,
    SUM(conteo) AS total_elegibles,
    SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) AS asignadas,
    SUM(CASE WHEN flag_declinado = 1 THEN conteo ELSE 0 END) AS no_asignadas,
    ROUND(100.0 * SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) / SUM(conteo), 2) AS pct_asignacion
FROM cubo
WHERE producto = ?
GROUP BY campania
ORDER BY campania
"""

SQL_DECLINE_REASONS = """
SELECT
    causa_no_asignacion,
    SUM(conteo) AS clientes,
    ROUND(100.0 * SUM(conteo) / (SELECT SUM(conteo) FROM cubo WHERE producto = ? AND flag_declinado = 1), 2) AS pct_del_total
FROM cubo
WHERE producto = ? AND flag_declinado = 1 AND causa_no_asignacion IS NOT NULL
GROUP BY causa_no_asignacion
ORDER BY clientes DESC
"""


class Question2Request(BaseModel):
    producto: Optional[str] = None


@router.get("/q2/params")
async def question_2_params():
    return JSONResponse(content={
        "producto": get_categorical_values("producto"),
    })


@router.post("/q2/ask")
async def question_2_ask(req: Question2Request):
    producto = req.producto or "CPC"

    assignment_data = run_query_as_text(SQL_ASSIGNMENT_BY_CAMPAIGN, (producto,))
    decline_data = run_query_as_text(SQL_DECLINE_REASONS, (producto, producto))

    data_context = f"""DATOS REALES PARA PRODUCTO {producto}:

PORCENTAJE DE ASIGNACIÓN POR CAMPAÑA:
{assignment_data}

CAUSAS DE NO ASIGNACIÓN:
{decline_data}
"""

    system_context = f"""Eres un analista experto en productos de crédito y campañas NPV.
Responde en español de manera ejecutiva.

PREGUNTA DE REFERENCIA: {SAMPLE_QUESTION}

EJEMPLO DE RESPUESTA ESPERADA:
{SAMPLE_ANSWER}

Usa los DATOS REALES para generar una respuesta similar para el producto {producto}.
Incluye: porcentaje actual, evolución por campaña, contexto y análisis de la tendencia.
"""

    from .llm_routes import model_manager, data_analyzer
    if not data_analyzer.is_loaded:
        from config import CHATBOT_CSV_PATH
        data_analyzer.load_data(CHATBOT_CSV_PATH)

    answer, inference_time = model_manager.generate_response(
        user_message=f"¿Qué porcentaje de las ofertas se asignó para {producto}?",
        data_context=system_context + "\n\n" + data_context,
    )

    return JSONResponse(content={
        "response": answer,
        "question_id": 2,
        "producto": producto,
        "inference_time": inference_time,
        "data_context": data_context,
    })
