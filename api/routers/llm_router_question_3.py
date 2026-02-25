"""
Router for Question 3: ¿Por qué no se asignaron el 100% de las ofertas de [producto]?

Analyzes decline reasons per product across campaigns.
Compatible with Python 3.9.20.
"""

import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .base_question_router import run_query_as_text, get_categorical_values

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/questions", tags=["question_3"])

SAMPLE_QUESTION = "¿Por qué no se asignaron el 100% de las ofertas de CPC? (Aplica para todos los productos)"

SAMPLE_ANSWER = """Esta es una pregunta fundamental, y la respuesta resulta particularmente relevante porque el problema es de naturaleza completamente interna.
En la campaña C5-2025, no se asignaron 226,942 ofertas de CPC. El hallazgo más significativo es que el 100% de estas ofertas no asignadas se perdieron por una única causa: "Business Rules assigned to revolving".
Las implicaciones de esto son las siguientes. Existen clientes que:
Superaron todos los filtros de elegibilidad inicial
Cumplen con el perfil crediticio requerido para un Crédito Personal al Consumo
Satisfacen todos los requisitos del producto
Han sido validados como candidatos apropiados
Sin embargo, en la etapa final del proceso, justo antes de asignar la oferta, una regla de negocio interna determina: "Este cliente será asignado a un producto revolvente (como una tarjeta de crédito)".
Este patrón se ha mantenido consistente a lo largo de las tres campañas:
C3-2025: 202,288 ofertas no asignadas por esta regla
C4-2025: 221,329 ofertas no asignadas por esta regla
C5-2025: 226,942 ofertas no asignadas por esta regla
No solo es un patrón consistente, sino que se está intensificando en términos absolutos.
Las ofertas se están perdiendo debido a un conflicto entre las propias reglas internas. Se trata de un problema de priorización de productos que requiere resolución urgente."""

SQL_DECLINE_BY_CAMPAIGN = """
SELECT
    campania,
    causa_no_asignacion,
    SUM(conteo) AS clientes_no_asignados
FROM cubo
WHERE producto = ? AND flag_declinado = 1 AND causa_no_asignacion IS NOT NULL
GROUP BY campania, causa_no_asignacion
ORDER BY campania, clientes_no_asignados DESC
"""

SQL_DECLINE_SUMMARY = """
SELECT
    causa_no_asignacion,
    SUM(conteo) AS total_no_asignados,
    ROUND(100.0 * SUM(conteo) / (SELECT SUM(conteo) FROM cubo WHERE producto = ? AND flag_declinado = 1), 2) AS pct_del_total,
    COUNT(DISTINCT campania) AS campanias_afectadas
FROM cubo
WHERE producto = ? AND flag_declinado = 1 AND causa_no_asignacion IS NOT NULL
GROUP BY causa_no_asignacion
ORDER BY total_no_asignados DESC
"""

SQL_TOTAL_NOT_ASSIGNED = """
SELECT
    campania,
    SUM(conteo) AS total_no_asignadas,
    SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) AS total_asignadas_camp
FROM cubo
WHERE producto = ?
GROUP BY campania
ORDER BY campania
"""


class Question3Request(BaseModel):
    producto: Optional[str] = None


@router.get("/q3/params")
async def question_3_params():
    return JSONResponse(content={
        "producto": get_categorical_values("producto"),
    })


@router.post("/q3/ask")
async def question_3_ask(req: Question3Request):
    producto = req.producto or "CPC"

    decline_by_camp = run_query_as_text(SQL_DECLINE_BY_CAMPAIGN, (producto,))
    decline_summary = run_query_as_text(SQL_DECLINE_SUMMARY, (producto, producto))
    totals = run_query_as_text(SQL_TOTAL_NOT_ASSIGNED, (producto,))

    data_context = f"""DATOS REALES PARA PRODUCTO {producto}:

RESUMEN DE CAUSAS DE NO ASIGNACIÓN (TODAS LAS CAMPAÑAS):
{decline_summary}

DETALLE POR CAMPAÑA:
{decline_by_camp}

TOTALES POR CAMPAÑA:
{totals}
"""

    system_context = f"""Eres un analista experto en productos de crédito y campañas NPV.
IMPORTANTE: SIEMPRE responde en español, sin importar en qué idioma se haga la pregunta.
Responde con análisis profundo de las causas.

PREGUNTA DE REFERENCIA: {SAMPLE_QUESTION}

EJEMPLO DE RESPUESTA ESPERADA:
{SAMPLE_ANSWER}

Usa los DATOS REALES para generar una respuesta similar para el producto {producto}.
Incluye: número de ofertas no asignadas, causas principales, tendencia entre campañas, e implicaciones.
"""

    from .llm_routes import model_manager, data_analyzer
    if not data_analyzer.is_loaded:
        from config import CHATBOT_CSV_PATH
        data_analyzer.load_data(CHATBOT_CSV_PATH)

    answer, inference_time = model_manager.generate_response(
        user_message=f"¿Por qué no se asignaron el 100% de las ofertas de {producto}?",
        data_context=system_context + "\n\n" + data_context,
    )

    return JSONResponse(content={
        "response": answer,
        "question_id": 3,
        "producto": producto,
        "inference_time": inference_time,
        "data_context": data_context,
    })
