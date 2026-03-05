'''Satriani API — AI Document Intelligence Platform'''
# Authors: Ivan Dario Penaloza Rojas

import os
import sys
from contextlib import asynccontextmanager
from config import *
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import subprocess
import logging
import re

# 0.3 getting routes
from routers import llm_routes
from routers import rag_routes
from routers import auth_routes
from routers import admin_routes
from routers import chat_routes

# Set JAVA_HOME and HADOOP_HOME
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

os.environ["SPARK_HOME"] = SPARK_HOME
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYLIB"] = os.environ["SPARK_HOME"] + "/python/lib"
sys.path.insert(0, os.environ["PYLIB"] + "/py4j-0.10.9.5-src.zip")
sys.path.insert(0, os.environ["PYLIB"] + "/pyspark.zip")

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
    os.environ["PATH"] = FFMPEG_WINDOWS_PATH + os.environ.get("PATH", "")
else:
    os.environ["FFMPEG_BINARY"] = FFMPEG_BINARY_LINUX_PATH
    os.environ["PATH"] = FFMPEG_LINUX_PATH + os.environ.get("PATH", "")


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    """Startup: init DB, load LLM model + RAG embeddings. Shutdown: cleanup."""
    # Initialize SQLite database
    import db as database
    database.init_db(SQLITE_DB_PATH, SESSION_SECRET, ADMIN_DEFAULT_USERNAME, ADMIN_DEFAULT_PASSWORD)
    logging.info("SQLite database initialized.")

    from routers.llm_routes import model_manager

    logging.info("Loading default LLM model...")
    model_manager.load_model()
    logging.info("Chatbot initialization complete.")

    # Initialize RAG: pre-load embedding model (but do NOT auto-ingest)
    from routers.rag_routes import get_all_rag_files, get_embedding_fn
    logging.info("Initializing RAG embedding model...")
    try:
        get_embedding_fn()
    except Exception as e:
        logging.warning("Could not load embedding model: %s", e)

    # Register existing files in SQLite (metadata only, no ingestion)
    rag_files = get_all_rag_files()
    for fpath in rag_files:
        fname = os.path.basename(fpath)
        if not database.get_document(fname):
            size_kb = round(os.path.getsize(fpath) / 1024, 1)
            database.register_document(fname, size_kb, "system")
    logging.info("RAG initialization complete. %d documents registered (ingestion is manual via admin panel).", len(rag_files))

    yield
    logging.info("Shutting down...")


app = FastAPI(
    title="Satriani API",
    description="AI Document Intelligence Platform",
    version="3.0.0",
    lifespan=lifespan,
)

app.include_router(auth_routes.router)
app.include_router(admin_routes.router)
app.include_router(chat_routes.router)
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
    """Serve the landing page."""
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get('/api/health')
async def health_check():
    return {"status": "healthy", "version": "3.0.0", "api_version": "v1"}


if __name__ == '__main__':
    if 'Desktop' in os.getcwd():
        uvicorn.run("main:app", host='localhost', port=PORT,
                     log_level="info", reload=True)
    elif is_local:
        uvicorn.run("main:app", host='0.0.0.0', port=PORT,
                     log_level="info", reload=True)
    else:
        uvicorn.run("main:app", host=FQDN, port=PORT,
                     log_level="info", reload=True,
                     ssl_keyfile=os.path.expanduser(SSL_AUTO_KEYFILE),
                     ssl_certfile=os.path.expanduser(SSL_AUTO_CERTIFICATE))
