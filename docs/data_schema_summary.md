# Data Schema Summary - cubo_datos_v2.csv

## Overview
This document provides a comprehensive understanding of the credit product data structure to guide chatbot query design and question answering capabilities.

**Dataset Size:** 248,308 rows × 15 columns

---

## Column Definitions & Schema

| Column | Type | Unique Values | Missing % | Definition |
|--------|------|---------------|-----------|------------|
| **etiqueta_grupo** | string | 2 | 0% | Customer classification: "elegibles" (prospect clients) or "mailbase" (clients receiving credit offers) |
| **producto** | string | 8 | 0% | Credit product type: CLI, DC, TCC, AAC, CPC, CNC, CPT, CNT |
| **toques** | numeric | 10 | 5.46% | Number of offer attempts (0-9) before product acceptance |
| **overlap_inicial** | string | 93 | 0% | Initial credit product(s) at start of Overlap process |
| **asignacion_final** | string | 118 | 0% | Final credit product(s) assigned after Overlap process |
| **ds_testlab** | string | 638 | 9.42% | Test groups to which customer belongs |
| **escenario** | string | 5 | 0% | Business strategy/rule for product offers |
| **conteo** | numeric | 9,130 | 0% | Count of customers with same characteristics |
| **linea_ofrecida** | numeric | 36,648 | 0% | Total credit line amount offered |
| **npv** | numeric | 130,943 | 0% | Net Present Value (projected) |
| **rentabilidad** | numeric | 117,458 | 8.32% | Projected profitability |
| **rr** | numeric | 100,333 | 6.84% | Projected RR metric |
| **campania** | string | 9 | 0% | Campaign identifier |
| **flag_declinado** | numeric | 2 | 0% | Binary: 0=offered, 1=declined |
| **causa_no_asignacion** | string | 10 | 35.23% | Reason for not offering product |

---

## Categorical Values Reference

### Customer Segmentation (etiqueta_grupo)
- **elegibles**: 151,472 records (61%)
- **mailbase**: 96,836 records (39%)

### Product Types (producto)
Distribution of credit products:
- **DC**: 77,264 (31.1%)
- **AAC**: 45,721 (18.4%)
- **CLI**: 42,501 (17.1%)
- **CPC**: 33,575 (13.5%)
- **CNC**: 24,600 (9.9%)
- **CPT**: 8,372 (3.4%)
- **CNT**: 8,307 (3.3%)
- **TCC**: 7,968 (3.2%)

### Business Scenarios (escenario)
- **4.NPV**: Most common strategy
- **5.Reglas_Negocio**: Business rules
- **3.Test**: Test scenarios
- **1.MVP**: Minimum viable product
- **2.CT_TECH**: Technical control

### Campaign Identifiers (campania)
- **C3-2024**: Primary campaign
- Additional campaigns: C3-2023, C2-2024, C2-2023, C1-2024, C1-2023, C4-2023, C4-2024, C5-2023

### Decline Reasons (causa_no_asignacion)
Top reasons for not offering products:
1. **Payment capacity** - Most common
2. **Prioritization of CLI CG**
3. **Business Rules assigned to revolving**
4. **Prioritization of LOP RS**
5. **Prioritization of PIL RS**
6. **Assigned to Top Ups**
7. **CLI CG for low utilization**
8. **Reglas Negocio**
9. **CLI CG for random lines**
10. **Prioritization of PIL and LOP RS**

---

## Numeric Statistics

### Offer Attempts (toques)
- Range: 0-9 attempts
- Average: 4.25 attempts
- Distribution: Relatively uniform across 0-9

### Customer Count (conteo)
- Range: 1 to 2,099,314 customers
- Median: 13 customers per characteristic group
- Mean: 1,357 customers

### Credit Lines (linea_ofrecida)
- Range: $0 to $71.8B
- Median: $1.35M
- Mean: $127.2M

### Financial Metrics
- **NPV**: Range from -$1.67M to $234M, Median: $2,569
- **Rentabilidad**: Range from -$326M to $20.2B, Median: $126K
- **RR**: Range from 0.00016 to 45,505, Median: 0.507

### Decline Rate
- **22%** of offers were declined (flag_declinado = 1)
- **78%** of offers were made to customers

---

## Data Quality Notes

1. **Missing Values:**
   - `causa_no_asignacion`: 35.23% missing (expected - only populated when declined)
   - `ds_testlab`: 9.42% missing
   - `rentabilidad`: 8.32% missing
   - `rr`: 6.84% missing
   - `toques`: 5.46% missing

2. **Complex Product Combinations:**
   - `overlap_inicial` has 93 unique combinations
   - `asignacion_final` has 118 unique combinations
   - Products are often combined with hyphens (e.g., "CPC-DC-CLI-AAC-")

3. **Test Groups:**
   - 638 unique test configurations in `ds_testlab`
   - Often combined with pipe separators (e.g., "C3_LOP_OO|C3_CG_CLI")

---

## Query Design Implications

### Recommended Query Types for Chatbot

1. **Segmentation Queries:**
   - "How many customers are in the mailbase vs elegibles group?"
   - "What's the distribution of products by customer segment?"

2. **Product Performance:**
   - "Which product has the highest NPV?"
   - "What's the average profitability by product type?"
   - "Show me the top 5 products by credit line offered"

3. **Campaign Analysis:**
   - "What's the decline rate for campaign C3-2024?"
   - "Compare NPV across different campaigns"
   - "Which campaign had the most offers?"

4. **Decline Analysis:**
   - "What are the top reasons for declining offers?"
   - "What percentage of offers were declined due to payment capacity?"
   - "Which products have the highest decline rate?"

5. **Business Strategy:**
   - "How many customers were assigned under NPV strategy?"
   - "Compare performance of Test vs MVP scenarios"
   - "What's the success rate by business scenario?"

6. **Overlap Analysis:**
   - "What are the most common initial product combinations?"
   - "How do products change from initial to final assignment?"
   - "What's the conversion rate from overlap to final assignment?"

7. **Financial Metrics:**
   - "What's the total NPV by product?"
   - "Calculate average credit line by customer segment"
   - "Show profitability trends by number of touches"

8. **Test Performance:**
   - "Which test groups have the best conversion rates?"
   - "Compare NPV across different test configurations"

---

## Technical Notes for Query Implementation

### Aggregation Patterns
- Use `conteo` for customer counts
- Sum `linea_ofrecida`, `npv`, `rentabilidad`, `rr` for totals
- Group by `producto`, `escenario`, `campania`, `etiqueta_grupo` for segmentation

### Filtering Patterns
- `flag_declinado = 0` for successful offers
- `flag_declinado = 1` for declined offers
- Filter by `campania` for time-based analysis
- Use `escenario` for strategy comparison

### Join Considerations
- Product combinations in `overlap_inicial` and `asignacion_final` may need string parsing
- Test groups in `ds_testlab` use pipe separators for multiple tests

---

**Generated:** For chatbot query design and data understanding
**Source:** api/static/data/cubo_datos_v2.csv
**Variable Definitions:** docs/context/variable_definitions.md
