'''Stargazer API'''
# Authors: Ivan Dario Penaloza Rojas <ip70574@citi.com>
# Manager: Ivan Dario Penaloza Rojas <ip70574@citi.com>

# 0 Configuration
import os
import sys
from contextlib import asynccontextmanager
# 0.1 Vivaldi dependencies
from config import *
# 0.2 Python dependencies
from fastapi import (
    FastAPI,
    Request
)
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import subprocess
import pyarrow.dataset as ds
import pyarrow.fs
import logging
import pandas as pd
import re
# 0.3 getting routes
from routers import llm_routes
from routers import rag_routes

# Set JAVA_HOME and HADOOP_HOME as before
os.environ['JAVA_HOME'] = JAVA_HOME
os.environ['HADOOP_HOME'] = HADOOP_HOME
try:
    classpath = subprocess.check_output(
        [os.path.join(os.environ['HADOOP_HOME'], 'bin/hadoop'),
         'classpath', '--glob'],
        universal_newlines=True
    ).strip()
    os.environ['CLASSPATH'] = classpath
except Exception:
    pass
os.environ['ARROW_LIBHDFS_DIR'] = ARROW_LIBHDFS_DIR

# configuration spark
os.environ["SPARK_HOME"] = SPARK_HOME
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON

os.environ["PYLIB"] = os.environ["SPARK_HOME"] + "/python/lib"
sys.path.insert(0, os.environ["PYLIB"] + "/py4j-0.10.9.5-src.zip")
sys.path.insert(0, os.environ["PYLIB"] + "/pyspark.zip")

# configuration kerberos mlflow
p = subprocess.Popen("whoami", stdout=subprocess.PIPE, shell=True)
USER_KERBEROS: str = p.communicate()[0].decode("utf-8")[:-1]
try:
    p = subprocess.Popen("klist", stdout=subprocess.PIPE, shell=True)
    output = p.communicate()[0].decode("utf-8")
    match = re.search(r'FILE:(/tmp/krb5cc_\d+)', output)
    KERBEROS_TICKET_CACHE = match.group(1)

    os.environ['MLFLOW_KERBEROS_TICKET_CACHE'] = KERBEROS_TICKET_CACHE
    os.environ['KRB5CCNAME'] = KERBEROS_TICKET_CACHE
except Exception:
    KERBEROS_TICKET_CACHE = ''

os.environ['MLFLOW_KERBEROS_TICKET_CACHE'] = KERBEROS_TICKET_CACHE
os.environ['MLFLOW_KERBEROS_USER'] = USER_KERBEROS
os.environ['KRB5CCNAME'] = KERBEROS_TICKET_CACHE

is_local = 'Desktop' in os.getcwd() or 'ProjectPrometheus' in os.getcwd()

if is_local:
    # Add FFmpeg binary directory to PATH WINDOWS
    os.environ["PATH"] = FFMPEG_WINDOWS_PATH + os.environ.get("PATH", "")
else:
    # Add FFmpeg binary directory to PATH LINUX
    os.environ["FFMPEG_BINARY"] = FFMPEG_BINARY_LINUX_PATH
    os.environ["PATH"] = FFMPEG_LINUX_PATH + os.environ.get("PATH", "")

p = subprocess.Popen("whoami", stdout=subprocess.PIPE, shell=True)
SOEID: str = p.communicate()[0].decode("utf-8")[:-1]

# Global data storage
app_data = {
    'blue_data': None,
    'green_data': None,
    'red_data': None,
    'filesystem': None
}

def read_parquet_to_pandas(path, filesystem):
    """Read parquet files from HDFS path and convert to pandas DataFrame"""
    try:
        # List all files in the directory
        file_info = filesystem.get_file_info(
            pyarrow.fs.FileSelector(path, recursive=True)
        )
        parquet_files = [
            f.path for f in file_info
            if f.is_file and f.path.endswith('.parquet')
        ]
        if not parquet_files:
            print(f"No parquet files found in {path}")
            return None
        # Create dataset and convert to pandas
        dataset = ds.dataset(
            parquet_files,
            filesystem=filesystem,
            format="parquet"
        )
        table = dataset.to_table()
        df = table.to_pandas()
        print(f"Successfully read {len(parquet_files)} parquet files from {path}")
        print(f"DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading parquet files from {path}: {e}")
        return None

# ─── Lifespan (replaces deprecated on_event) ─────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    """Startup: load data + chatbot model + RAG embeddings. Shutdown: cleanup."""
    global app_data

    # 1) Load Chatbot (dataset + default model)
    from routers.llm_routes import data_analyzer, model_manager

    logging.info("Loading chatbot dataset...")
    data_analyzer.load_data(CHATBOT_CSV_PATH)

    logging.info("Loading default LLM model...")
    model_manager.load_model()
    logging.info("Chatbot initialization complete.")

    # 2) Initialize RAG: ingest .docx files if not already done
    from routers.rag_routes import get_all_docx_files, ingest_docx_file, get_embedding_fn
    logging.info("Initializing RAG embedding model...")
    get_embedding_fn()  # warm up
    docx_files = get_all_docx_files()
    for fpath in docx_files:
        try:
            result = ingest_docx_file(fpath)
            logging.info("RAG ingest: %s → %s (%d chunks)", result["filename"], result["status"], result["chunks"])
        except Exception as e:
            logging.warning("RAG ingest failed for %s: %s", fpath, e)
    logging.info("RAG initialization complete.")

    yield  # app is running

    logging.info("Shutting down...")


app = FastAPI(
    title="Satriani API",
    description="RAG Document Chat powered by offline LLMs",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(llm_routes.router)
app.include_router(rag_routes.router)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')
templates.env.globals.update(
    all_lst=all_lst,
    neg_all_lst=neg_all_lst
)

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(FAVICON_PATH)

@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    """Redirect root to Satriani RAG chat page"""
    return RedirectResponse(url='/api/v1/rag/chat')

@app.get('/api/health')
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "api_version": "v1"
    }

if __name__ == '__main__':
    if 'Desktop' in os.getcwd():
        # Windows / Desktop local dev
        uvicorn.run(
            "main:app",
            host='localhost',
            port=PORT,
            log_level="info",
            reload=True
        )
    elif is_local:
        # Linux local dev (e.g. ProjectPrometheus) — HTTP, no SSL
        uvicorn.run(
            "main:app",
            host='0.0.0.0',
            port=PORT,
            log_level="info",
            reload=True
        )
    else:
        # Remote / production — HTTPS with SSL certs
        uvicorn.run(
            "main:app",
            host=FQDN,
            port=PORT,
            log_level="info",
            reload=True,
            ssl_keyfile=os.path.expanduser(SSL_AUTO_KEYFILE),
            ssl_certfile=os.path.expanduser(SSL_AUTO_CERTIFICATE)
        )