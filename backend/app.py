from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import json
import subprocess
import docker
from docker.types import DeviceRequest
import time
import os
import requests
import redis.asyncio as redis
from datetime import datetime
from contextlib import asynccontextmanager
import logging




# print(f'** connecting to redis on port: {os.getenv("REDIS_PORT")} ... ')
r = redis.Redis(host="redis", port=int(os.getenv("REDIS_PORT", 6379)), db=0)

LOG_PATH= './logs'
LOGFILE_CONTAINER = f'{LOG_PATH}/logfile_{os.getenv("CONTAINER_BACKEND")}.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.INFO, format='[%(asctime)s - %(name)s - %(levelname)s - %(message)s]')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] started logging in {LOGFILE_CONTAINER}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

print(f' %%%%% trying to start docker ...')
client = docker.from_env()
print(f' %%%%% docker started!')
print(f' %%%%% trying to docker network ...')
network_name = "sys_net"
# try:
#     network = client.networks.get(network_name)
# except docker.errors.NotFound:
#     network = client.networks.create(network_name, driver="bridge")
# print(f' %%%%% docker network started! ...')




@app.get("/")
async def root():
    return f'Hello from backend server {os.getenv("BACKEND_PORT")}!'

@app.post("/docker")
async def fndocker(request: Request):
    try:
        req_data = await request.json()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] req_data > {req_data}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] req_data > {req_data}')
        
        
        if req_data["method"] == "list":
            res_container_list = client.containers.list(all=True)
            return JSONResponse([container.attrs for container in res_container_list])

    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": f'{e}'})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=f'{os.getenv("BACKEND_IP")}', port=int(os.getenv("BACKEND_PORT")))