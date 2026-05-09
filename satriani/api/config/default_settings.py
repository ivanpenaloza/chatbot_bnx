'''Configuration file Stargazer API'''
# Authors: Ivan Dario Penaloza Rojas <ip70574@citi.com>
# Manager: Ivan Dario Penaloza Rojas <ip70574@citi.com>

'''
    DO NOT TOUCH
'''

import os
import socket

PORT = int(os.environ.get('PORT'))
FQDN = socket.gethostname()
FAVICON_PATH = "images/banamex.ico"
JAVA_HOME = "/usr/java/default"
HADOOP_HOME ='/opt/cloudera/parcels/CDH-7.1.9-1.cdh7.1.9.p1054.69605909/lib/hadoop'
ARROW_LIBHDFS_DIR = '/opt/cloudera/parcels/CDH-7.1.9-1.cdh7.1.9.p1054.69605909/lib64'
SPARK_HOME = "/opt/cloudera/parcels/SPARK3/lib/spark3"
PYSPARK_PYTHON = "/opt/cloudera/parcels/citiconda39-9.9.46/bin/python"
FFMPEG_WINDOWS_PATH = r"C:\Users\IP70574\Downloads\ffmpeg-2025-09-18-git-c373636f55-full_build\ffmpeg-2025-09-18-git-c373636f55-full_build\bin;"
FFMPEG_LINUX_PATH = "/data/1/gcgamdlmxpysp/ip70574/ffmpeg-7.0.2-amd64-static:"
FFMPEG_BINARY_LINUX_PATH = "/data/1/gcgamdlmxpysp/ip70574/ffmpeg-7.0.2-amd64-static/ffmpeg"
BLUE_PARQUET_PATH = '/data/gcgaacqmxpysp/work/hive/gcgaacqmxpysp_work/jm34240/atm_opt_terceros/implementation/test_1/dashboard_inputs/atm_data_202506'
GREEN_PARQUET_PATH = '/data/gcgaacqmxpysp/work/hive/gcgaacqmxpysp_work/jm34240/atm_opt_terceros/implementation/test_1/dashboard_inputs/geogenius_score_merchants_202506'
RED_PARQUET_PATH = '/data/gcgaacqmxpysp/work/hive/gcgaacqmxpysp_work/jm34240/atm_opt_terceros/implementation/test_1/dashboard_inputs/ubicajeros_data_202506'
HDFS_HOST_NAME = 'bdqtr006x11h3.lac.nsroot.net'
HDFS_HOST_PORT = 8020
SSL_AUTO_KEYFILE = "/data/1/gcgamdlmxpysp/ip70574/certificates_geogenius_api/server.key"
SSL_AUTO_CERTIFICATE = "/data/1/gcgamdlmxpysp/ip70574/certificates_geogenius_api/server.crt"