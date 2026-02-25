"""
Router for Question 7: Preguntas de contexto (¿Qué es NPV?, ¿Qué es CPC?, ¿Qué significa X?)

Contextual/definitional questions about terms, products, and business rules.
Compatible with Python 3.9.20.
"""

import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .base_question_router import run_query_as_text, get_categorical_values

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/questions", tags=["question_7"])

SAMPLE_QUESTION = "Preguntas de contexto: ¿Qué es NPV? ¿Qué es CPC? ¿Qué significa 'Business Rules assigned to revolving'?"

CONTEXT_DEFINITIONS = """
DEFINICIONES DE CONTEXTO:

NPV (Net Present Value / Valor Presente Neto):
Es una métrica financiera que calcula el valor que un cliente genera para la institución a lo largo del tiempo, expresado en términos de valor presente. Toma todos los flujos futuros (ingresos menos costos como fondeo, operativos, pérdida esperada) y los descuenta a valor presente.

PRODUCTOS:
- CPC (Crédito Personal al Consumo): Préstamo personal de monto fijo, no revolvente. Se entrega en una sola exhibición y se amortiza en cuotas fijas.
- DC (Disponible): Producto de crédito disponible para clientes existentes.
- AAC (Add a Card): Oferta de tarjeta adicional para clientes existentes, menores costos de adquisición.
- TCC (Tarjeta de Crédito Citi): Tarjeta de crédito estándar, producto revolvente.
- CNC (Crédito Nómina Citi): Crédito para clientes con nómina domiciliada, menor riesgo.
- CLI (Credit Line Increase): Aumento de línea de crédito para clientes existentes.
- CPT (Crédito Personal Total): Variante de crédito personal.
- CNT (Crédito Nómina Total): Variante de crédito nómina.

CAUSAS DE NO ASIGNACIÓN:
- "Business Rules assigned to revolving": Regla que prioriza productos revolventes sobre no revolventes. Afecta principalmente a CPC.
- "Payment capacity": Capacidad de pago insuficiente del cliente.
- "Prioritization of CLI CG": Priorización de aumento de línea de crédito.
- "Prioritization of LOP RS": Priorización de línea de operación.
- "Assigned to Top Ups": Cliente asignado a incrementos de producto existente.
- "CLI CG for low utilization": Aumento de línea para clientes con baja utilización.
- "Reglas Negocio": Reglas de negocio generales.

VARIABLES DEL DATASET:
- etiqueta_grupo: "elegibles" (prospectos) o "mailbase" (clientes que reciben oferta)
- conteo: Número de clientes con mismas características
- linea_ofrecida: Suma de líneas de crédito ofrecidas
- rentabilidad: Rentabilidad proyectada
- rr: Tasa de respuesta proyectada
- escenario: Estrategia de negocio (1.MVP, 2.CT_TECH, 3.Test, 4.NPV, 5.Reglas_Negocio)
- flag_declinado: 0=oferta asignada, 1=oferta no asignada
- overlap_inicial / asignacion_final: Productos al inicio y final del proceso de Overlap
"""

SQL_PRODUCT_SUMMARY = """
SELECT
    producto,
    SUM(conteo) AS total_clientes,
    ROUND(SUM(npv) / NULLIF(SUM(conteo), 0), 2) AS npv_promedio,
    ROUND(100.0 * SUM(CASE WHEN flag_declinado = 0 THEN conteo ELSE 0 END) / SUM(conteo), 2) AS pct_asignacion
FROM cubo
GROUP BY producto
ORDER BY total_clientes DESC
"""

SQL_DECLINE_REASONS_ALL = """
SELECT
    causa_no_asignacion,
    SUM(conteo) AS total_afectados,
    COUNT(DISTINCT producto) AS productos_afectados
FROM cubo
WHERE flag_declinado = 1 AND causa_no_asignacion IS NOT NULL
GROUP BY causa_no_asignacion
ORDER BY total_afectados DESC
"""


class Question7Request(BaseModel):
    subtopic: Optional[str] = None  # e.g. "NPV", "CPC", "Business Rules assigned to revolving"


@router.get("/q7/params")
async def question_7_params():
    causas = get_categorical_values("causa_no_asignacion")
    productos = get_categorical_values("producto")
    subtopics = ["NPV", "Overlap", "Escenarios"] + productos + causas
    return JSONResponse(content={
        "subtopic": subtopics,
    })


@router.post("/q7/ask")
async def question_7_ask(req: Question7Request):
    subtopic = req.subtopic or "NPV"

    product_summary = run_query_as_text(SQL_PRODUCT_SUMMARY)
    decline_reasons = run_query_as_text(SQL_DECLINE_REASONS_ALL)

    data_context = f"""DATOS DE REFERENCIA:

RESUMEN POR PRODUCTO:
{product_summary}

CAUSAS DE NO ASIGNACIÓN:
{decline_reasons}
"""

    system_context = f"""Eres un analista experto en productos de crédito y campañas NPV.
Responde en español de manera clara y didáctica.

{CONTEXT_DEFINITIONS}

El usuario pregunta sobre: "{subtopic}"

Proporciona una explicación clara y contextualizada usando las definiciones anteriores y los datos reales.
Si es un producto, incluye sus métricas. Si es una causa de no asignación, explica su impacto.
"""

    from .llm_routes import model_manager, data_analyzer
    if not data_analyzer.is_loaded:
        from config import CHATBOT_CSV_PATH
        data_analyzer.load_data(CHATBOT_CSV_PATH)

    answer, inference_time = model_manager.generate_response(
        user_message=f"¿Qué es {subtopic}? Explícame en el contexto de las campañas de crédito.",
        data_context=system_context + "\n\n" + data_context,
    )

    return JSONResponse(content={
        "response": answer,
        "question_id": 7,
        "subtopic": subtopic,
        "inference_time": inference_time,
        "data_context": data_context,
    })
