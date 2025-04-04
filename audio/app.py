import uvicorn
import redis.asyncio as redis
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import os
from faster_whisper import WhisperModel
import pynvml
import torch

audio_model = None
def load_audio(req_model_size):
    try:
        global audio_model
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_audio] trying to start WhisperModel with size: {req_model_size}')
        if audio_model is None:
            audio_model = WhisperModel(req_model_size, device="cpu", compute_type="int8")
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_audio] [success] WhisperModel started!')
    
    except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_audio] [error] Failed to load WhisperModel')
            raise

def transcribe_audio(req_model_size,audio_file_path):
    try:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] trying to load WhisperModel req_model_size: {req_model_size} ...')
        load_audio(req_model_size)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] WhisperModel loaded!')
        
        start_time = time.time()
        
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] trying to transcribe path: {audio_file_path} ...')
        segments, info = audio_model.transcribe(audio_file_path)
        full_text = "\n".join([segment.text for segment in segments])
        processing_time = time.time() - start_time
        
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] finished transcribing audio from {audio_file_path}! lang found: {info.language} len text_length: {len(full_text)} in {processing_time:.2f}s ...')
        
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] returning ...')
        
        return f"Detected language: {info.language}\n\n{full_text}\n\nProcessing time: {processing_time:.2f}s"
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] [error]: {e}')
        return f"Error: {e}"


redis_connection = None

def start_redis(req_redis_port):
    try:
        r = redis.Redis(host="redis", port=req_redis_port, db=0)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [start_redis] Redis started successfully.')
        return r
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [start_redis] Failed to start Redis on port {req_redis_port}: {e}')
        raise


def initialize_nvml():
    try:
        pynvml.nvmlInit()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [initialize_nvml] NVML initialized successfully.')
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [initialize_nvml] Failed to initialize NVML: {e}')

initialize_nvml()


def cuda_support_bool():
    try:
        res_cuda_support = torch.cuda.is_available()
        if res_cuda_support:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [cuda_support_bool] CUDA found')
        else:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [cuda_support_bool] CUDA not supported!')
        return res_cuda_support
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [cuda_support_bool] {e}')
        return False


def cuda_device_count():
    try:
        res_gpu_int = torch.cuda.device_count()
        res_gpu_int_arr = [i for i in range(res_gpu_int)]
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [cuda_support_bool] {str(res_gpu_int)}x GPUs found {res_gpu_int_arr}')
        return res_gpu_int_arr
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [cuda_device_count] Failed to get device_count. Using default [0]. Error: {e}')
        return [0]

gpu_int_arr = cuda_device_count()




app = FastAPI()
llm_instance = None
             
@app.get("/")
async def root():
    return f'Hello from audio server!'
             
@app.get("/t")
async def fntest():
    res_transcribe = transcribe_audio("small","nk.mp3")
    return f'{res_transcribe}'

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=f'{os.getenv("AUDIO_IP")}', port=int(os.getenv("AUDIO_PORT")))
    

