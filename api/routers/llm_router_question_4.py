"""
Router for Question 4: ¿Cuál es el NPV promedio de [producto]?

NPV analysis per product with comparative context.
Compatible with Python 3.9.20.
"""

import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .base_question_router import run_query_as_text, get_categorical_values

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/questions", tags=["question_4"])

SAMPLE_QUESTION = "¿Cuál es el NPV promedio de CPC? (Aplica para todos los productos)"

SAMPLE_ANSWER = """Con base en los análisis, el NPV (Net Present Value o Valor Presente Neto) promedio para CPC es de $2,847 por cliente.
Este valor se encuentra en el rango medio del portafolio de productos de crédito al consumo. El NPV de CPC presenta las siguientes características:
Rango de valores:
NPV mínimo: $1,200
NPV promedio: $2,847
NPV máximo: $8,500
Desviación estándar: $1,456
Contexto comparativo con otros productos:
CNC tiene el NPV unitario más alto del portafolio con $4,567
AAC registra un NPV de $3,456, superior a CPC
TCC presenta un NPV de $2,145, ligeramente inferior a CPC
DC tiene el NPV más bajo con $1,234
NPV Total proyectado para CPC:
Considerando el volumen de 2,600,405 ofertas asignadas en C5-2025, el NPV total proyectado para CPC es de aproximadamente $7,403 millones."""

SQL_NPV_BY_PRODUCT = """
SELECT
    producto,
    ROUND(SUM(npv) / NULLIF(SUM(conteo), 0), 2) AS npv_promedio_por_cliente,
    ROUND(SUM(npv), 2) AS npv_total,
    SUM(conteo) AS total_clientes,
    ROUND(MIN(npv * 1.0 / NULLIF(conteo, 0)), 2) AS npv_min_unitario,
    ROUND(MAX(npv * 1.0 / NULLIF(conteo, 0)), 2) AS npv_max_unitario
FROM cubo
WHERE flag_declinado = 0
GROUP BY producto
ORDER BY npv_promedio_por_cliente DESC
"""

SQL_NPV_PRODUCT_BY_CAMPAIGN = """
SELECT
    campania,
    SUM(conteo) AS clientes_asignados,
    ROUND(SUM(npv), 2) AS npv_total,
    ROUND(SUM(npv) / NULLIF(SUM(conteo), 0), 2) AS npv_promedio
FROM cubo
WHERE producto = ? AND flag_declinado = 0
GROUP BY campania
ORDER BY campania
"""

SQL_NPV_STATS = """
SELECT
    ROUND(SUM(npv) / NULLIF(SUM(conteo), 0), 2) AS npv_promedio,
    ROUND(SUM(npv), 2) AS npv_total,
    SUM(conteo) AS total_clientes_asignados
FROM cubo
WHERE producto = ? AND flag_declinado = 0
"""


class Question4Request(BaseModel):
    producto: Optional[str] = None


@router.get("/q4/params")
async def question_4_params():
    return JSONResponse(content={
        "producto": get_categorical_values("producto"),
    })


@router.post("/q4/ask")
async def question_4_ask(req: Question4Request):
    producto = req.producto or "CPC"

    all_products = run_query_as_text(SQL_NPV_BY_PRODUCT)
    by_campaign = run_query_as_text(SQL_NPV_PRODUCT_BY_CAMPAIGN, (producto,))
    stats = run_query_as_text(SQL_NPV_STATS, (producto,))

    data_context = f"""DATOS REALES PARA PRODUCTO {producto}:

NPV ESTADÍSTICAS DEL PRODUCTO:
{stats}

NPV POR CAMPAÑA PARA {producto}:
{by_campaign}

COMPARATIVO NPV TODOS LOS PRODUCTOS (solo asignados):
{all_products}
"""

    system_context = f"""Eres un analista experto en productos de crédito y campañas NPV.
IMPORTANTE: SIEMPRE responde en español, sin importar en qué idioma se haga la pregunta.
Responde con análisis financiero detallado.

PREGUNTA DE REFERENCIA: {SAMPLE_QUESTION}

EJEMPLO DE RESPUESTA ESPERADA:
{SAMPLE_ANSWER}

Usa los DATOS REALES para generar una respuesta similar para el producto {producto}.
Incluye: NPV promedio, rango, comparativo con otros productos, NPV total proyectado.
Nota: npv y conteo son sumas agregadas por grupo de características. NPV promedio por cliente = SUM(npv)/SUM(conteo).
"""

    from .llm_routes import model_manager, data_analyzer
    if not data_analyzer.is_loaded:
        from config import CHATBOT_CSV_PATH
        data_analyzer.load_data(CHATBOT_CSV_PATH)

    answer, inference_time = model_manager.generate_response(
        user_message=f"¿Cuál es el NPV promedio de {producto}?",
        data_context=system_context + "\n\n" + data_context,
    )

    return JSONResponse(content={
        "response": answer,
        "question_id": 4,
        "producto": producto,
        "inference_time": inference_time,
        "data_context": data_context,
    })
