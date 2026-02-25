"""
Router for Question 9: ¿Cuáles son las principales estrategias que mueven los resultados de Overlap?

Overlap analysis: product conflicts, prioritization strategies, and recommendations.
Compatible with Python 3.9.20.
"""

import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .base_question_router import run_query_as_text, get_categorical_values

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/questions", tags=["question_9"])

SAMPLE_QUESTION = "¿Cuáles son las principales estrategias que mueven los resultados de Overlap?"

SAMPLE_ANSWER = """El análisis de Overlap (superposición de ofertas entre productos) es fundamental para comprender la dinámica de asignación del portafolio.
Overlap se refiere a clientes que califican simultáneamente para múltiples productos. Tipos principales:
1. Overlap Revolvente vs. No Revolvente: Clientes que califican para revolventes (tarjetas) y no revolventes (CPC, CNC)
2. Overlap Intra-Segmento: Clientes que califican para múltiples productos de la misma categoría
3. Overlap de Expansión: Clientes existentes que califican para productos nuevos y expansión

Estrategia de Priorización de Productos Revolventes:
- Regla "Business Rules assigned to revolving" aplicada en etapa final
- CPC principal afectado: 226,942 clientes redirigidos en C5-2025
- Tendencia creciente: 12.2% en términos absolutos

Recomendaciones:
1. Unificar funnel de priorización al INICIO del proceso
2. Segmentar gestión de exposición en AAC por tiers de riesgo
3. Revisar balance CPC-Revolvente con test A/B
4. Programa de domiciliación para expandir CNC

Impacto total de optimizar: 400-500K ofertas adicionales, $1,500-2,100M en NPV adicional."""

SQL_OVERLAP_PATTERNS = """
SELECT
    overlap_inicial,
    COUNT(*) AS registros,
    SUM(conteo) AS total_clientes,
    ROUND(SUM(npv), 2) AS npv_total
FROM cubo
GROUP BY overlap_inicial
ORDER BY total_clientes DESC
LIMIT 20
"""

SQL_OVERLAP_VS_FINAL = """
SELECT
    overlap_inicial,
    asignacion_final,
    SUM(conteo) AS clientes,
    ROUND(SUM(npv), 2) AS npv
FROM cubo
WHERE flag_declinado = 0
GROUP BY overlap_inicial, asignacion_final
ORDER BY clientes DESC
LIMIT 20
"""

SQL_REVOLVING_CONFLICT = """
SELECT
    campania,
    producto,
    SUM(conteo) AS clientes_redirigidos
FROM cubo
WHERE flag_declinado = 1
    AND causa_no_asignacion = 'Business Rules assigned to revolving'
GROUP BY campania, producto
ORDER BY campania, clientes_redirigidos DESC
"""

SQL_DECLINE_BY_STRATEGY = """
SELECT
    escenario,
    causa_no_asignacion,
    SUM(conteo) AS total_no_asignados
FROM cubo
WHERE flag_declinado = 1 AND causa_no_asignacion IS NOT NULL
GROUP BY escenario, causa_no_asignacion
ORDER BY total_no_asignados DESC
LIMIT 20
"""

SQL_PRODUCT_OVERLAP_COUNT = """
SELECT
    producto,
    COUNT(DISTINCT overlap_inicial) AS combinaciones_overlap,
    SUM(conteo) AS total_clientes,
    ROUND(100.0 * SUM(CASE WHEN flag_declinado = 1 THEN conteo ELSE 0 END) / SUM(conteo), 2) AS pct_no_asignacion
FROM cubo
GROUP BY producto
ORDER BY pct_no_asignacion DESC
"""


class Question9Request(BaseModel):
    campania: Optional[str] = None


@router.get("/q9/params")
async def question_9_params():
    return JSONResponse(content={
        "campania": get_categorical_values("campania"),
    })


@router.post("/q9/ask")
async def question_9_ask(req: Question9Request):
    campanias = get_categorical_values("campania")
    campania = req.campania or (campanias[-1] if campanias else "C5-2025")

    overlap_patterns = run_query_as_text(SQL_OVERLAP_PATTERNS)
    overlap_vs_final = run_query_as_text(SQL_OVERLAP_VS_FINAL)
    revolving_conflict = run_query_as_text(SQL_REVOLVING_CONFLICT)
    decline_by_strategy = run_query_as_text(SQL_DECLINE_BY_STRATEGY)
    product_overlap = run_query_as_text(SQL_PRODUCT_OVERLAP_COUNT)

    data_context = f"""DATOS REALES DE OVERLAP:

PATRONES DE OVERLAP INICIAL MÁS FRECUENTES:
{overlap_patterns}

OVERLAP INICIAL VS ASIGNACIÓN FINAL (top 20):
{overlap_vs_final}

CONFLICTO REVOLVENTE POR CAMPAÑA Y PRODUCTO:
{revolving_conflict}

NO ASIGNACIÓN POR ESTRATEGIA Y CAUSA:
{decline_by_strategy}

RESUMEN DE OVERLAP POR PRODUCTO:
{product_overlap}
"""

    system_context = f"""Eres un analista experto en productos de crédito y campañas NPV.
Responde en español con análisis estratégico profundo sobre Overlap.

PREGUNTA DE REFERENCIA: {SAMPLE_QUESTION}

EJEMPLO DE RESPUESTA ESPERADA:
{SAMPLE_ANSWER}

Usa los DATOS REALES para generar una respuesta similar.
Incluye: definición de Overlap, tipos, estrategias principales, impacto cuantificado, y recomendaciones.
"""

    from .llm_routes import model_manager, data_analyzer
    if not data_analyzer.is_loaded:
        from config import CHATBOT_CSV_PATH
        data_analyzer.load_data(CHATBOT_CSV_PATH)

    answer, inference_time = model_manager.generate_response(
        user_message="¿Cuáles son las principales estrategias que mueven los resultados de Overlap?",
        data_context=system_context + "\n\n" + data_context,
    )

    return JSONResponse(content={
        "response": answer,
        "question_id": 9,
        "campania": campania,
        "inference_time": inference_time,
        "data_context": data_context,
    })
