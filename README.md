
# StargazerAPI

The fourth project from **Citilabs** designed for:

1. ATMs' locations and insights
2. Banamex' merchants, their locations and insights
   

See also 
- [FastAPI](https://fastapi.tiangolo.com/) for web development documentation


### About this project

- 📦 A basic [setup.py](setup.py) file to provide installation, packaging and distribution for your project.  
  Template uses setuptools because it's the de-facto standard for Python packages, you can run `make switch-to-poetry` later if you want.
- 🤖 A [pyproject.toml](pyproject.toml) file that provides details about dependencies and Stargazer's Ivan Dario Penaloza Rojas

> Curious about architectural views and patterns? read [Stargazer](https://cedt-gct-confluence.nam.nsroot.net/confluence/display/MEXAA/Sherlock++V2.0)  
> If you want to contribute to this project please open an [issue](https://cedt-gct-bitbucket.nam.nsroot.net/bitbucket/users/ip70574/repos/stargazerapi/pull-requests) or fork and send a PULL REQUEST.

[❤️ Sponsor Citilabs](https://cedt-gct-confluence.nam.nsroot.net/confluence/display/MEXAA/Citilabs)

---
# Usage

## INSTALL StargazerAPI AND RUN IT USING EXISTING PYTHON ENVIRONMENT

```bash
ssh bdqtr005x17h3.lac.nsroot.net
kinit
bash
cd /data/1/gcgamdlmxpysp/ip70574
source envs/env39/bin/activate
cd /data/1/gcgamdlmxpysp/
mkdir YOUR_PERSONAL_FOLDER_NAME
cd YOUR_PERSONAL_FOLDER_NAME
git clone https://cedt-gct-bitbucket.nam.nsroot.net/bitbucket/scm/~ip70574/stargazerapi.git
cd stargazerapi
git checkout develop
cd api  
PORT='5050' python main.py & 
(press CTRL+D to disconnect console and not affect StargazerAPI)
```

## INSTALL StargazerAPI AND RUN IT USING A NEW PYTHON ENVIRONMENT (IDEAL FOR DEPLOYMENT)

```bash
ssh bdqtr005x17h3.lac.nsroot.net
kinit
bash
cd /data/1/gcgamdlmxpysp/
mkdir YOUR_PERSONAL_FOLDER_NAME
cd YOUR_PERSONAL_FOLDER_NAME
git clone https://cedt-gct-bitbucket.nam.nsroot.net/bitbucket/scm/~ip70574/stargazerapi.git
cd stargazerapi
git checkout develop
python39 -m venv env39
source env39/bin/activate
pip install -r requirements.txt
cd api  
PORT='5050' python main.py & 
(press CTRL+D to disconnect console and not affect StargazerAPI)
```


## RUN StargazerAPI

> **YOU CAN CHOOSE WHATEVER PORT YOU WANT** Choose a port between 5000 and 9999.

```bash
cd /data/1/gcgamdlmxpysp/YOUR_PERSONAL_FOLDER_NAME/stargazer/api
PORT='5050' python main.py & 
(press CTRL+D to disconnect console and not affect StargazerAPI)
```

## HOW TO OPEN StargazerAPI

Just do CTRL + CLICK on the link in your console

```bash
http://bdqtr005x17h3.lac.nsroot.net:PORT
```

## HOW TO QUIT StargazerAPI

There are different ways to quit StargazerAPI:

**Alternative 1:** DO USE CTRL + C from console in the directory where the api is located. This will cause the uvicorn web server to stop StargazerAPI.

**Alternative 2:** If the previous step does not work. Use the following commands: 
 ```bash
kill -9 $(lsof -t -i :PORT)
```
Where PORT must be your port number.