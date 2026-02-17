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
    """Startup: load data + chatbot model. Shutdown: cleanup."""
    global app_data

    # 1) Load Stargazer data (local CSV or datalake)
    if is_local:
        try:
            logging.info("Running locally - loading data from CSV files...")
            csv_base_path = os.path.join(
                os.path.dirname(__file__),
                "static", "data"
            )
            blue_csv_path = os.path.join(csv_base_path, "blue_data.csv")
            green_csv_path = os.path.join(csv_base_path, "green_data.csv")
            red_csv_path = os.path.join(csv_base_path, "red_data.csv")

            if os.path.exists(blue_csv_path):
                app_data['blue_data'] = pd.read_csv(blue_csv_path)
                logging.info(f"Blue data loaded. Shape: {app_data['blue_data'].shape}")
            else:
                logging.warning(f"Blue CSV not found: {blue_csv_path}")

            if os.path.exists(green_csv_path):
                app_data['green_data'] = pd.read_csv(green_csv_path)
                logging.info(f"Green data loaded. Shape: {app_data['green_data'].shape}")
            else:
                logging.warning(f"Green CSV not found: {green_csv_path}")

            if os.path.exists(red_csv_path):
                app_data['red_data'] = pd.read_csv(red_csv_path)
                logging.info(f"Red data loaded. Shape: {app_data['red_data'].shape}")
            else:
                logging.warning(f"Red CSV not found: {red_csv_path}")

            logging.info("Local CSV data loading completed!")
        except Exception as e:
            logging.error(f"Error loading CSV data: {e}")
            app_data = {'blue_data': None, 'green_data': None, 'red_data': None, 'filesystem': None}
    else:
        try:
            logging.info("Running remotely - loading data from datalake...")
            app_data['filesystem'] = pyarrow.fs.HadoopFileSystem(
                HDFS_HOST_NAME, port=HDFS_HOST_PORT,
                user=SOEID, kerb_ticket=KERBEROS_TICKET_CACHE
            )
            app_data['blue_data'] = read_parquet_to_pandas(BLUE_PARQUET_PATH, app_data['filesystem'])
            app_data['green_data'] = read_parquet_to_pandas(GREEN_PARQUET_PATH, app_data['filesystem'])
            app_data['red_data'] = read_parquet_to_pandas(RED_PARQUET_PATH, app_data['filesystem'])
            logging.info("All datalake data loaded successfully!")
        except Exception as e:
            logging.error(f"Error loading datalake data: {e}")
            app_data = {'blue_data': None, 'green_data': None, 'red_data': None, 'filesystem': None}

    # 2) Load Chatbot (dataset + Gemma 3 offline model)
    from routers.llm_routes import data_analyzer, gemma_client

    logging.info("Loading chatbot dataset...")
    data_analyzer.load_data(CHATBOT_CSV_PATH)

    logging.info("Loading Gemma 3 offline model...")
    gemma_client.load_model()
    logging.info("Chatbot initialization complete.")

    yield  # app is running

    logging.info("Shutting down...")


app = FastAPI(
    title="Stargazer API",
    description="ATM Location Intelligence HUB API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(llm_routes.router)

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
    """Redirect root to chatbot page"""
    return RedirectResponse(url='/api/v1/chatbot')

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