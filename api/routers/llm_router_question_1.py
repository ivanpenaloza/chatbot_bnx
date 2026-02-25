"""
Router for Question 1: ¿Cómo va la campaña? / ¿Cuáles fueron los resultados de la campaña?

Provides campaign overview with assignment efficiency per product.
Compatible with Python 3.9.20.
"""

import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .base_question_router import run_query_as_text, get_categorical_values

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/questions", tags=["question_1"])

SAMPLE_QUESTION = "¿Cómo va la campaña? / ¿Cuáles fueron los resultados de la campaña?"

SAMPLE_ANSWER = """Los resultados de la campaña C5-2025 muestran un panorama mixto dependiendo del producto que se analice.
En términos generales, se lograron asignar 11.26 millones de ofertas con una eficiencia del 91.6%, lo cual es aceptable. Aquí hay un poco más de detalle por producto:
CPC tuvo un desempeño complicado. Se asignaron 2.6 millones de ofertas con una tasa del 92%, lo cual representa una ligera recuperación respecto a la campaña anterior, pero se sigue viendo una tendencia a la baja en volumen. El problema principal aquí es que se están perdiendo más de 226 mil ofertas porque entran en conflicto con las reglas de productos revolventes.
DC es el mejor producto en términos de eficiencia. Se logró un 95.3% de asignación con casi 1.85 millones de ofertas. Este producto tiene muy poca fricción en las reglas de negocio y está funcionando muy bien.
AAC es interesante porque es el producto con mayor volumen - se asignaron 3.42 millones de ofertas - pero tiene la eficiencia más baja del portafolio, apenas 88.7%. Aquí el problema es que está chocando con límites de exposición crediticia.
TCC está en un punto medio con 2.16 millones de ofertas asignadas y 91.2% de eficiencia. Aquí se ve un fenómeno de migración - muchos clientes que podrían tener una tarjeta clásica están siendo desviados a productos premium.
CNC es el producto más eficiente después de DC, con 94.5% de asignación y 1.23 millones de ofertas.
En resumen: hay productos que funcionan muy bien (DC, CNC), uno con gran volumen pero desafíos operativos (AAC), y otros que necesitan atención estratégica urgente (CPC, TCC)."""


# ─── SQL Queries ─────────────────────────────────────────────────────────────

SQL_OVERALL = """
SELECT
    campania,
    SUM(conteo) AS total_clientes,
    SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) AS asignadas,
    SUM(CASE WHEN flag_declinado = 1 THEN conteo ELSE 0 END) AS no_asignadas,
    ROUND(100.0 * SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) / SUM(conteo), 2) AS pct_asignacion
FROM cubo
WHERE campania = ?
GROUP BY campania
"""

SQL_BY_PRODUCT = """
SELECT
    producto,
    SUM(conteo) AS total_clientes,
    SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) AS asignadas,
    SUM(CASE WHEN flag_declinado = 1 THEN conteo ELSE 0 END) AS no_asignadas,
    ROUND(100.0 * SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) / SUM(conteo), 2) AS pct_asignacion,
    ROUND(SUM(npv), 2) AS npv_total,
    ROUND(SUM(npv) * 1.0 / NULLIF(SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END), 0), 2) AS npv_promedio
FROM cubo
WHERE campania = ?
GROUP BY producto
ORDER BY asignadas DESC
"""

SQL_TOP_DECLINE_REASONS = """
SELECT
    producto,
    causa_no_asignacion,
    SUM(conteo) AS clientes_no_asignados
FROM cubo
WHERE campania = ? AND flag_declinado = 1 AND causa_no_asignacion IS NOT NULL
GROUP BY producto, causa_no_asignacion
ORDER BY clientes_no_asignados DESC
LIMIT 15
"""


class Question1Request(BaseModel):
    campania: Optional[str] = None


@router.get("/q1/params")
async def question_1_params():
    """Return available parameter values for question 1."""
    return JSONResponse(content={
        "campania": get_categorical_values("campania"),
    })


@router.post("/q1/ask")
async def question_1_ask(req: Question1Request):
    """Answer question 1 using pre-defined SQL queries and LLM context."""
    campanias = get_categorical_values("campania")
    campania = req.campania or (campanias[-1] if campanias else "C5-2025")

    # Run queries
    overall_data = run_query_as_text(SQL_OVERALL, (campania,))
    product_data = run_query_as_text(SQL_BY_PRODUCT, (campania,))
    decline_data = run_query_as_text(SQL_TOP_DECLINE_REASONS, (campania,))

    # Build context for LLM
    data_context = f"""DATOS REALES DE LA CAMPAÑA {campania}:

RESUMEN OVERALL:
{overall_data}

DESGLOSE POR PRODUCTO:
{product_data}

PRINCIPALES CAUSAS DE NO ASIGNACIÓN:
{decline_data}
"""

    system_context = f"""Eres un analista experto en productos de crédito y campañas NPV.
Responde en español de manera ejecutiva y con insights accionables.

PREGUNTA DE REFERENCIA: {SAMPLE_QUESTION}

EJEMPLO DE RESPUESTA ESPERADA (para otra campaña):
{SAMPLE_ANSWER}

Usa los DATOS REALES proporcionados para generar una respuesta similar pero con los números correctos de la campaña {campania}.
Incluye: resumen general, desglose por producto con eficiencia, problemas identificados y conclusión.
"""

    # Import model manager from llm_routes
    from .llm_routes import model_manager, data_analyzer
    if not data_analyzer.is_loaded:
        from config import CHATBOT_CSV_PATH
        data_analyzer.load_data(CHATBOT_CSV_PATH)

    answer, inference_time = model_manager.generate_response(
        user_message=f"¿Cómo va la campaña {campania}? Dame los resultados detallados.",
        data_context=system_context + "\n\n" + data_context,
        dynamic_context=""
    )

    return JSONResponse(content={
        "response": answer,
        "question_id": 1,
        "campania": campania,
        "inference_time": inference_time,
        "data_context": data_context,
    })
