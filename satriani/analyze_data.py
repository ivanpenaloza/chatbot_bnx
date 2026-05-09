"""
Descriptive Analytics for cubo_datos_v2.csv
This script provides comprehensive analysis of the credit product data including:
- Schema information
- Statistical summaries
- Unique values for categorical columns
- Data quality checks
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Variable definitions from documentation
VARIABLE_DEFINITIONS = {
    'etiqueta_grupo': 'Customer classification: "potenciales" (prospect clients) or "mailbase" (clients receiving credit offers)',
    'producto': 'Credit product type indicator (e.g., CLI, DC, TCC)',
    'toques': 'Cumulative number of offer attempts made to the customer before product acceptance',
    'overlap_inicial': 'Initial credit product(s) with which the customer starts the Overlap process',
    'asignacion_final': 'Final credit product(s) assigned to the customer after the Overlap process',
    'ds_testlab': 'Tests to which the customer belongs',
    'escenario': 'Business strategy or rule under which product offers were defined',
    'conteo': 'Total number of customers sharing the same characteristics (product, scenario, test, etc.)',
    'linea_ofrecida': 'Total sum of credit line amounts offered to all customers within the same characteristics',
    'npv': 'Sum of projected Net Present Value (NPV) for customers with the same characteristics',
    'rentabilidad': 'Sum of projected profitability for customers with the same characteristics',
    'rr': 'Sum of projected RR for customers with the same characteristics',
    'campania': 'Campaign indicator',
    'flag_declinado': 'Binary indicator showing whether the product was offered to the customer or not',
    'causa_no_asignacion': 'Description specifying why the product offer was not made to the customer (e.g., payment capacity)'
}

def load_data(file_path):
    """Load the CSV data"""
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✓ Data loaded successfully: {len(df):,} rows, {len(df.columns)} columns\n")
    return df

def analyze_schema(df):
    """Analyze and display schema information"""
    print("=" * 80)
    print("DATA SCHEMA")
    print("=" * 80)
    
    schema_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        schema_info.append({
            'Column': col,
            'Data Type': dtype,
            'Null Count': null_count,
            'Null %': f"{null_pct:.2f}%",
            'Unique Values': unique_count,
            'Definition': VARIABLE_DEFINITIONS.get(col, 'N/A')
        })
    
    schema_df = pd.DataFrame(schema_info)
    print(schema_df.to_string(index=False))
    print()
    return schema_df

def analyze_numeric_columns(df):
    """Provide statistical summary for numeric columns"""
    print("=" * 80)
    print("NUMERIC COLUMNS - STATISTICAL SUMMARY")
    print("=" * 80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        stats = df[numeric_cols].describe().T
        stats['missing'] = df[numeric_cols].isnull().sum()
        stats['missing_pct'] = (stats['missing'] / len(df)) * 100
        print(stats.to_string())
    else:
        print("No numeric columns found")
    print()

def analyze_categorical_columns(df):
    """Display all unique values for string/categorical columns"""
    print("=" * 80)
    print("CATEGORICAL COLUMNS - UNIQUE VALUES")
    print("=" * 80)
    
    # Identify categorical columns (object dtype or low cardinality)
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 50:
            categorical_cols.append(col)
    
    for col in categorical_cols:
        unique_values = df[col].dropna().unique()
        value_counts = df[col].value_counts(dropna=False)
        
        print(f"\n{col.upper()}")
        print("-" * 80)
        print(f"Definition: {VARIABLE_DEFINITIONS.get(col, 'N/A')}")
        print(f"Total unique values: {len(unique_values)}")
        print(f"Missing values: {df[col].isnull().sum()} ({(df[col].isnull().sum()/len(df)*100):.2f}%)")
        print(f"\nValue distribution:")
        print(value_counts.to_string())
        print()

def analyze_data_quality(df):
    """Analyze data quality issues"""
    print("=" * 80)
    print("DATA QUALITY ANALYSIS")
    print("=" * 80)
    
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    print(f"\nMissing values by column:")
    
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_data) > 0:
        print(missing_data.to_string(index=False))
    else:
        print("No missing values found!")
    print()

def generate_summary_report(df):
    """Generate a comprehensive summary report"""
    print("=" * 80)
    print("SUMMARY INSIGHTS")
    print("=" * 80)
    
    print(f"\n1. CUSTOMER SEGMENTATION:")
    if 'etiqueta_grupo' in df.columns:
        print(df['etiqueta_grupo'].value_counts().to_string())
    
    print(f"\n2. PRODUCT DISTRIBUTION:")
    if 'producto' in df.columns:
        print(df['producto'].value_counts().to_string())
    
    print(f"\n3. CAMPAIGN ANALYSIS:")
    if 'campania' in df.columns:
        print(df['campania'].value_counts().to_string())
    
    print(f"\n4. DECLINED OFFERS:")
    if 'flag_declinado' in df.columns:
        declined_stats = df['flag_declinado'].value_counts()
        print(declined_stats.to_string())
        if 1 in declined_stats.index:
            declined_pct = (declined_stats[1] / len(df)) * 100
            print(f"\nDeclined rate: {declined_pct:.2f}%")
    
    print(f"\n5. TOP DECLINE REASONS:")
    if 'causa_no_asignacion' in df.columns:
        print(df['causa_no_asignacion'].value_counts().head(10).to_string())
    
    print(f"\n6. BUSINESS METRICS SUMMARY:")
    metrics = ['conteo', 'linea_ofrecida', 'npv', 'rentabilidad', 'rr']
    for metric in metrics:
        if metric in df.columns:
            total = df[metric].sum()
            avg = df[metric].mean()
            print(f"  {metric}: Total={total:,.2f}, Average={avg:,.2f}")
    
    print()

def main():
    """Main execution function"""
    # File path
    data_file = Path("api/static/data/cubo_datos_v2.csv")
    
    if not data_file.exists():
        print(f"Error: File not found at {data_file}")
        return
    
    # Load data
    df = load_data(data_file)
    
    # Run analyses
    analyze_schema(df)
    analyze_numeric_columns(df)
    analyze_categorical_columns(df)
    analyze_data_quality(df)
    generate_summary_report(df)
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nThis analysis provides insights into the nature of the data for future")
    print("chatbot query design. Key areas to consider:")
    print("  • Customer segmentation (etiqueta_grupo)")
    print("  • Product performance (producto, npv, rentabilidad)")
    print("  • Campaign effectiveness (campania, flag_declinado)")
    print("  • Decline analysis (causa_no_asignacion)")
    print("  • Test scenarios (ds_testlab, escenario)")

if __name__ == "__main__":
    main()
