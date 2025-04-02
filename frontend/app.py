from dataclasses import dataclass, fields
import gradio as gr
import redis
import threading
import time
import os
import requests
import json
import subprocess
import sys
import ast
import time
from datetime import datetime
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import huggingface_hub
from huggingface_hub import snapshot_download
import logging
import psutil



def transcribe_audio(audio_file):
    req_file = audio_file
    return f'req_file: {req_file}'



REQUEST_TIMEOUT = 300
def wait_for_backend(backend_url, timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.post(backend_url, json={"req_method": "list"}, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                print("Backend container is online.")
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass  # Backend is not yet reachable
        time.sleep(5)  # Wait for 5 seconds before retrying
    print(f"Timeout: Backend container did not come online within {timeout} seconds.")
    return False



docker_container_list = []
current_models_data = []
db_gpu_data = []
db_gpu_data_len = ''
GLOBAL_SELECTED_MODEL_ID = ''
GLOBAL_MEM_TOTAL = 0
GLOBAL_MEM_USED = 0
GLOBAL_MEM_FREE = 0

try:
    r = redis.Redis(host="redis", port=6379, db=0)
    db_gpu = json.loads(r.get('db_gpu'))
    # print(f'db_gpu: {db_gpu} {len(db_gpu)}')
    db_gpu_data_len = len(db_gpu_data)
except Exception as e:
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')

LOG_PATH= './logs'
LOGFILE_CONTAINER = './logs/logfile_container_frontend.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.INFO, format='[%(asctime)s - %(name)s - %(levelname)s - %(message)s]')
logging.info(f' [START] started logging in {LOGFILE_CONTAINER}')

def load_log_file(req_container_name):
    print(f' **************** GOT LOG FILE REQUEST FOR CONTAINER ID: {req_container_name}')
    logging.info(f' **************** GOT LOG FILE REQUEST FOR CONTAINER ID: {req_container_name}')
    try:
        with open(f'{LOG_PATH}/logfile_{req_container_name}.log', "r", encoding="utf-8") as file:
            lines = file.readlines()
            last_20_lines = lines[-20:]
            reversed_lines = last_20_lines[::-1]
            return ''.join(reversed_lines)
    except Exception as e:
        return f'{e}'



DEFAULTS_PATH = "/usr/src/app/utils/defaults.json"
if not os.path.exists(DEFAULTS_PATH):
    logging.info(f' [START] File missing: {DEFAULTS_PATH}')

with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
    defaults_frontend = json.load(f)["frontend"]
    logging.info(f' [START] SUCCESS! Loaded: {DEFAULTS_PATH}')
    logging.info(f' [START] {len(defaults_frontend['vllm_supported_architectures'])} supported vLLM architectures found!')









def get_container_data():
    try:
        res_container_data_all = json.loads(r.get('db_container'))
        return res_container_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
   
def get_network_data():
    try:
        res_network_data_all = json.loads(r.get('db_network'))
        return res_network_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e

def get_gpu_data():
    try:
        res_gpu_data_all = json.loads(r.get('db_gpu'))
        return res_gpu_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e

def get_vllm_data():
    try:
        res_vllm_data_all = json.loads(r.get('db_vllm'))
        return res_vllm_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e

def get_disk_data():
    try:
        res_disk_data_all = json.loads(r.get('db_disk'))
        return res_disk_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e

def get_docker_container_list():
    global docker_container_list
    response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"list"})
    # print(f'[get_docker_container_list] response: {response}')
    res_json = response.json()
    # print(f'[get_docker_container_list] res_json: {res_json}')
    docker_container_list = res_json.copy()
    if response.status_code == 200:
        # print(f'[get_docker_container_list] res = 200')
        return res_json
    else:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        # logging.info(f'[get_docker_container_list] [get_docker_container_list] res_json: {res_json}')
        return f'Error: {response.status_code}'

def docker_api_logs(req_model):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"logs","req_model":req_model})
        res_json = response.json()
        return ''.join(res_json["result_data"])
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action stop'

def docker_api_network(req_container_name):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"network","req_container_name":req_container_name})
        res_json = response.json()
        if res_json["result"] == 200:
            return f'{res_json["result_data"]["networks"]["eth0"]["rx_bytes"]}'
        else:
            return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action network {res_json}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action network {e}'
    
def docker_api_start(req_model):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"start","req_model":req_model})
        res_json = response.json()
        return res_json
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action start {e}'

def docker_api_stop(req_model):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"stop","req_model":req_model})
        res_json = response.json()
        return res_json
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action stop {e}'

def docker_api_delete(req_model):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"delete","req_model":req_model})
        res_json = response.json()
        return res_json
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action delete {e}'

def docker_api_create(req_model, req_pipeline_tag, req_port_model, req_port_vllm):
    try:
        req_container_name = str(req_model).replace('/', '_')
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"create","req_container_name":req_container_name,"req_model":req_model,"req_runtime":"nvidia","req_port_model":req_port_model,"req_port_vllm":req_port_vllm})
        response_json = response.json()
        
        new_entry = [{
            "gpu": 8,
            "path": f'/home/cloud/.cache/huggingface/{req_model}',
            "container": "0",
            "container_status": "0",
            "running_model": req_container_name,
            "model": req_model,
            "pipeline_tag": req_pipeline_tag,
            "port_model": req_port_model,
            "port_vllm": req_port_vllm
        }]
        r.set("db_gpu", json.dumps(new_entry))

        print(response_json["result"])
        if response_json["result"] == 200:
            return f'{response_json["result_data"]}'
        else:
            return f'Create result ERR no container_id: {str(response_json)}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'error docker_api_create'

def search_models(query):
    try:
        global current_models_data    
        response = requests.get(f'https://huggingface.co/api/models?search={query}')
        response_models = response.json()
        current_models_data = response_models.copy()
        model_ids = [m["id"] for m in response_models]
        if len(model_ids) < 1:
            model_ids = ["No models found!"]
        return gr.update(choices=model_ids, value=response_models[0]["id"], show_label=True, label=f'found {len(response_models)} models!')
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')

def calculate_model_size(json_info): # to fix    
    try:
        d_model = json_info.get("hidden_size") or json_info.get("d_model")
        num_hidden_layers = json_info.get("num_hidden_layers", 0)
        num_attention_heads = json_info.get("num_attention_heads") or json_info.get("decoder_attention_heads") or json_info.get("encoder_attention_heads", 0)
        intermediate_size = json_info.get("intermediate_size") or json_info.get("encoder_ffn_dim") or json_info.get("decoder_ffn_dim", 0)
        vocab_size = json_info.get("vocab_size", 0)
        num_channels = json_info.get("num_channels", 3)
        patch_size = json_info.get("patch_size", 16)
        torch_dtype = json_info.get("torch_dtype", "float32")
        bytes_per_param = 2 if torch_dtype == "float16" else 4
        total_size_in_bytes = 0
        
        if json_info.get("model_type") == "vit":
            embedding_size = num_channels * patch_size * patch_size * d_model
            total_size_in_bytes += embedding_size

        if vocab_size and d_model:
            embedding_size = vocab_size * d_model
            total_size_in_bytes += embedding_size

        if num_attention_heads and d_model and intermediate_size:
            attention_weights_size = num_hidden_layers * (d_model * d_model * 3)
            ffn_weights_size = num_hidden_layers * (d_model * intermediate_size + intermediate_size * d_model)
            layer_norm_weights_size = num_hidden_layers * (2 * d_model)

            total_size_in_bytes += (attention_weights_size + ffn_weights_size + layer_norm_weights_size)

        if json_info.get("is_encoder_decoder"):
            encoder_size = num_hidden_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            decoder_layers = json_info.get("decoder_layers", 0)
            decoder_size = decoder_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            
            total_size_in_bytes += (encoder_size + decoder_size)

        return total_size_in_bytes * 2
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return 0


def get_info(selected_id):
    
    print(f' @@@ [get_info] 0')
    logging.info(f' @@@ [get_info] 0')   
    container_name = ""
    res_model_data = {
        "search_data" : "",
        "model_id" : "",
        "pipeline_tag" : "",
        "architectures" : "",
        "transformers" : "",
        "private" : "",
        "downloads" : ""
    }
    
    if selected_id == None:
        print(f' @@@ [get_info] selected_id NOT FOUND!! RETURN ')
        logging.info(f' @@@ [get_info] selected_id NOT FOUND!! RETURN ') 
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
    
    global current_models_data
    global GLOBAL_SELECTED_MODEL_ID
    GLOBAL_SELECTED_MODEL_ID = selected_id
    print(f' @@@ [get_info] {selected_id} 2')
    logging.info(f' @@@ [get_info] {selected_id} 2')  
    
    print(f' @@@ [get_info] {selected_id} 3')
    logging.info(f' @@@ [get_info] {selected_id} 3')  
    container_name = str(res_model_data["model_id"]).replace('/', '_')
    print(f' @@@ [get_info] {selected_id} 4')
    logging.info(f' @@@ [get_info] {selected_id} 4')  
    if len(current_models_data) < 1:
        print(f' @@@ [get_info] len(current_models_data) < 1! RETURN ')
        logging.info(f' @@@ [get_info] len(current_models_data) < 1! RETURN ') 
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
    try:
        print(f' @@@ [get_info] {selected_id} 5')
        logging.info(f' @@@ [get_info] {selected_id} 5') 
        for item in current_models_data:
            print(f' @@@ [get_info] {selected_id} 6')
            logging.info(f' @@@ [get_info] {selected_id} 6') 
            if item['id'] == selected_id:
                print(f' @@@ [get_info] {selected_id} 7')
                logging.info(f' @@@ [get_info] {selected_id} 7') 
                res_model_data["search_data"] = item
                
                if "pipeline_tag" in item:
                    res_model_data["pipeline_tag"] = item["pipeline_tag"]
  
                if "tags" in item:
                    if "transformers" in item["tags"]:
                        res_model_data["transformers"] = True
                    else:
                        res_model_data["transformers"] = False
                                    
                if "private" in item:
                    res_model_data["private"] = item["private"]
                                  
                if "architectures" in item:
                    res_model_data["architectures"] = item["architectures"][0]
                                                    
                if "downloads" in item:
                    res_model_data["downloads"] = item["downloads"]
                  
                container_name = str(res_model_data["model_id"]).replace('/', '_')
                
                print(f' @@@ [get_info] {selected_id} 8')
                logging.info(f' @@@ [get_info] {selected_id} 8') 
                
                return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
            else:
                
                print(f' @@@ [get_info] {selected_id} 9')
                logging.info(f' @@@ [get_info] {selected_id} 9') 
                
                return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
    except Exception as e:
        print(f' @@@ [get_info] {selected_id} 10')
        logging.info(f' @@@ [get_info] {selected_id} 10') 
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name

def get_additional_info(selected_id):    
        res_model_data = {
            "hf_data" : "",
            "hf_data_config" : "",
            "config_data" : "",
            "architectures" : "",
            "model_type" : "",
            "quantization" : "",
            "tokenizer_config" : "",
            "model_id" : selected_id,
            "size" : 0,
            "gated" : "",
            "torch_dtype" : "",
            "hidden_size" : "",
            "cuda_support" : "",
            "compute_capability" : ""
        }                
        try:
            try:
                model_info = huggingface_hub.model_info(selected_id)
                model_info_json = vars(model_info)
                res_model_data["hf_data"] = model_info_json
                
                if "config" in model_info.__dict__:
                    res_model_data['hf_data_config'] = model_info_json["config"]
                    if "architectures" in model_info_json["config"]:
                        res_model_data['architectures'] = model_info_json["config"]["architectures"][0]
                    if "model_type" in model_info_json["config"]:
                        res_model_data['model_type'] = model_info_json["config"]["model_type"]
                    if "tokenizer_config" in model_info_json["config"]:
                        res_model_data['tokenizer_config'] = model_info_json["config"]["tokenizer_config"]
                               
                if "gated" in model_info.__dict__:
                    res_model_data['gated'] = model_info_json["gated"]
                
                if "safetensors" in model_info.__dict__:
                    print(f'  FOUND safetensors')
                    logging.info(f'  GFOUND safetensors')   
                    
                    safetensors_json = vars(model_info.safetensors)
                    
                    
                    print(f'  FOUND safetensors:::::::: {safetensors_json}')
                    logging.info(f'  GFOUND safetensors:::::::: {safetensors_json}') 
                    try:
                        quantization_key = next(iter(safetensors_json['parameters'].keys()))
                        print(f'  FOUND first key in parameters:::::::: {quantization_key}')
                        res_model_data['quantization'] = quantization_key
                        
                    except Exception as get_model_info_err:
                        print(f'  first key NOT FOUND in parameters:::::::: {quantization_key}')
                        pass
                    
                    print(f'  FOUND safetensors TOTAL :::::::: {safetensors_json["total"]}')
                    logging.info(f'  GFOUND safetensors:::::::: {safetensors_json["total"]}')
                                        
                    print(f'  ooOOOOOOOOoooooo res_model_data["quantization"] {res_model_data["quantization"]}')
                    logging.info(f'ooOOOOOOOOoooooo res_model_data["quantization"] {res_model_data["quantization"]}')
                    if res_model_data["quantization"] == "F32":
                        print(f'  ooOOOOOOOOoooooo found F32 -> x4')
                        logging.info(f'ooOOOOOOOOoooooo found F32 -> x4')
                    else:
                        print(f'  ooOOOOOOOOoooooo NUUUH FIND F32 -> x2')
                        logging.info(f'ooOOOOOOOOoooooo NUUUH FIND F32 -> x2')
                        res_model_data['size'] = int(safetensors_json["total"]) * 2
                else:
                    print(f' !!!!DIDNT FIND safetensors !!!! :::::::: ')
                    logging.info(f' !!!!!! DIDNT FIND safetensors !!:::::::: ') 
            
            
            
            except Exception as get_model_info_err:
                res_model_data['hf_data'] = f'{get_model_info_err}'
                pass
                    
            try:
                url = f'https://huggingface.co/{selected_id}/resolve/main/config.json'
                response = requests.get(url)
                if response.status_code == 200:
                    response_json = response.json()
                    res_model_data["config_data"] = response_json
                    
                    if "architectures" in res_model_data["config_data"]:
                        res_model_data["architectures"] = res_model_data["config_data"]["architectures"][0]
                        
                    if "torch_dtype" in res_model_data["config_data"]:
                        res_model_data["torch_dtype"] = res_model_data["config_data"]["torch_dtype"]
                        print(f'  ooOOOOOOOOoooooo torch_dtype: {res_model_data["torch_dtype"]}')
                        logging.info(f'ooOOOOOOOOoooooo torch_dtype: {res_model_data["torch_dtype"]}')
                    if "hidden_size" in res_model_data["config_data"]:
                        res_model_data["hidden_size"] = res_model_data["config_data"]["hidden_size"]
                        print(f'  ooOOOOOOOOoooooo hidden_size: {res_model_data["hidden_size"]}')
                        logging.info(f'ooOOOOOOOOoooooo hidden_size: {res_model_data["hidden_size"]}')
                else:
                    res_model_data["config_data"] = f'{response.status_code}'
                    
            except Exception as get_config_json_err:
                res_model_data["config_data"] = f'{get_config_json_err}'
                pass                       
            
            if res_model_data["size"] == 0:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] **************** [get_additional_info] res_model_data["size"] == 0 ...')
                logging.info(f' **************** [get_additional_info] res_model_data["size"] == 0...')
                try:
                    res_model_data["size"] = calculate_model_size(res_model_data["config_data"]) 
                except Exception as get_config_json_err:
                    res_model_data["size"] = 0

            # quantization size 
            if res_model_data['quantization'] == "F32" or res_model_data["torch_dtype"] == "float32":
                res_model_data["size"] = res_model_data["size"] * 2
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] **************** res_model_data["size"] * 2 ...')
                logging.info(f' **************** res_model_data["size"] * 2...')
    
    
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["architectures"], res_model_data["model_id"], res_model_data["size"], res_model_data["gated"], res_model_data["model_type"], res_model_data["quantization"], res_model_data["torch_dtype"], res_model_data["hidden_size"]
        
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["model_id"], res_model_data["size"], res_model_data["gated"], res_model_data["model_type"],  res_model_data["quantization"], res_model_data["torch_dtype"], res_model_data["hidden_size"]

def gr_load_check(selected_model_id, selected_model_architectures, selected_model_pipeline_tag, selected_model_transformers, selected_model_size, selected_model_private, selected_model_gated, selected_model_model_type, selected_model_quantization):
    
    global GLOBAL_MEM_TOTAL
    global GLOBAL_MEM_USED
    global GLOBAL_MEM_FREE
    

    
    
    # check CUDA support mit backend call
    
    # if "gguf" in selected_model_id.lower():
    #     return f'Selected a GGUF model!', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    req_model_storage = "/models"
    req_model_path = f'{req_model_storage}/{selected_model_id}'
    
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] **************** [gr_load_check] searching {selected_model_id} in {req_model_storage} (req_model_path: {req_model_path}) ...')
    logging.info(f' **************** [gr_load_check] searching {selected_model_id} in {req_model_storage} (req_model_path: {req_model_path})...')
    


    models_found = []
    try:                   
        if os.path.isdir(req_model_storage):
            print(f' **************** found model storage path! {req_model_storage}')
            print(f' **************** getting folder elements ...')       
            logging.info(f' **************** found model storage path! {req_model_storage}')
            logging.info(f' **************** getting folder elements ...')                        
            for m_entry in os.listdir(req_model_storage):
                m_path = os.path.join(req_model_storage, m_entry)
                if os.path.isdir(m_path):
                    for item_sub in os.listdir(m_path):
                        sub_item_path = os.path.join(m_path, item_sub)
                        models_found.append(sub_item_path)        
            print(f' **************** found models ({len(models_found)}): {models_found}')
            logging.info(f' **************** found models ({len(models_found)}): {models_found}')
        else:
            print(f' **************** found models ({len(models_found)}): {models_found}')
            logging.info(f' **************** ERR model path not found! {req_model_storage}')
    except Exception as e:
        logging.info(f' **************** ERR getting models in {req_model_storage}: {e}')

    
    logging.info(f' **************** does requested model path match downloaded?')
    model_path = selected_model_id
    if req_model_path in models_found:
        print(f' **************** FOUND MODELS ALREADY!!! {selected_model_id} ist in {models_found}')
        model_path = req_model_path
        return f'Model already downloaded!', gr.update(visible=True), gr.update(visible=True)
    else:
        print(f' **************** NUH UH DIDNT FIND MODEL YET!! {selected_model_id} ist NAWT in {models_found}')
    
    
        
    if selected_model_architectures == '':
        return f'Selected model has no architecture', gr.update(visible=False), gr.update(visible=False)
    
    
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] **************** [gr_load_check] selected_model_architectures.lower() : {selected_model_architectures.lower()}')
    logging.info(f' **************** [gr_load_check] selected_model_architectures.lower() : {selected_model_architectures.lower()}')

    if selected_model_architectures.lower() not in defaults_frontend['vllm_supported_architectures']:
        if selected_model_transformers != 'True':   
            return f'Selected model architecture is not supported by vLLM but transformers are available (you may try to load the model in gradio Interface)', gr.update(visible=True), gr.update(visible=True)
        else:
            return f'Selected model architecture is not supported by vLLM and has no transformers', gr.update(visible=False), gr.update(visible=False)     
    
    if selected_model_pipeline_tag == '':
        return f'Selected model has no pipeline tag', gr.update(visible=True), gr.update(visible=True)
            
    if selected_model_pipeline_tag not in ["text-generation","automatic-speech-recognition"]:
        return f'Only "text-generation" and "automatic-speech-recognition" models supported', gr.update(visible=False), gr.update(visible=False)
    
    if selected_model_private != 'False':        
        return f'Selected model is private', gr.update(visible=False), gr.update(visible=False)
        
    if selected_model_gated != 'False':        
        return f'Selected model is gated', gr.update(visible=False), gr.update(visible=False)
        
    if selected_model_transformers != 'True':        
        return f'Selected model has no transformers', gr.update(visible=True), gr.update(visible=True)
        
    if selected_model_size == '0':        
        return f'Selected model has no size', gr.update(visible=False), gr.update(visible=False)




    
    # print(f' **>>> gr_load_check !! !! !! >> 0 >> {GLOBAL_MEM_TOTAL}')
    # logging.info(f' **>>> gr_load_check !! !! !! >> 0 >> {GLOBAL_MEM_TOTAL}')
    # print(f' **>>> gr_load_check !! !! !! >> 0 >> {GLOBAL_MEM_USED}')
    # logging.info(f' **>>> gr_load_check !! !! !! >> 0 >> {GLOBAL_MEM_USED}')
    # print(f' **>>> gr_load_check !! !! !! >> 0 >> {GLOBAL_MEM_FREE}')
    # logging.info(f' **>>> gr_load_check !! !! !! >> 0 >> {GLOBAL_MEM_FREE}')
    
    # if selected_model_id == '':
    #     return f'Model not found!', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    
    
    
    # print(f' HÄÄÄÄÄÄÄÄÄ bis hier oder net')
    # logging.info(f' HÄÄÄÄÄÄÄÄÄ bis hier oder net')
    
    
    
    
    
    # print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] ********* [gr_load_check] checking if enough memory size for selected model available ....  ...')
    # logging.info(f' ********* [gr_load_check] checking if enough memory size for selected model available .... ...')    
    
    # print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] ********* [gr_load_check] GLOBAL_MEM_TOTAL {GLOBAL_MEM_TOTAL}')
    # logging.info(f' ********* [gr_load_check] GLOBAL_MEM_TOTAL {GLOBAL_MEM_TOTAL} ')    
    # print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] ********* [gr_load_check] GLOBAL_MEM_USED {GLOBAL_MEM_USED}')
    # logging.info(f' ********* [gr_load_check] GLOBAL_MEM_USED {GLOBAL_MEM_USED} ')    
    # print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] ********* [gr_load_check] GLOBAL_MEM_FREE {GLOBAL_MEM_FREE}')
    # logging.info(f' ********* [gr_load_check] GLOBAL_MEM_FREE {GLOBAL_MEM_FREE} ')
    
 
    # print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] ********* [gr_load_check] selected_model_size {selected_model_size}')
    # logging.info(f' ********* [gr_load_check] selected_model_size {selected_model_size} ')
    
    
    # print(f' HÄÄÄÄÄÄÄÄÄ bis hier oder net 2')
    # logging.info(f' HÄÄÄÄÄÄÄÄÄ bis hier oder net 2')
    
    
    # # check model > size memory size
    # if float(selected_model_size) > (float(GLOBAL_MEM_TOTAL.split()[0])*1024**2):
    #     print(f' HÄÄÄÄÄÄÄÄÄ bis hier oder net 444')
    #     logging.info(f' HÄÄÄÄÄÄÄÄÄ bis hier oder net 444')
    #     return f'ERR: model size extends GPU memory! {float(selected_model_size)}/{(float(GLOBAL_MEM_TOTAL.split()[0])*1024**2)} ', gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    #     # return f'ERR: model size extends GPU memory!', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    # if float(selected_model_size) > (float(GLOBAL_MEM_FREE.split()[0])*1024**2):
    #     print(f' HÄÄÄÄÄÄÄÄÄ bis hier oder net 555')
    #     logging.info(f' HÄÄÄÄÄÄÄÄÄ bis hier oder net 555')
    #     return f'Please clear GPU memory! {float(selected_model_size)}/{(float(GLOBAL_MEM_FREE.split()[0])*1024**2)} ', gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    
    

    
    # print(f' HÄÄÄÄÄÄÄÄÄ bis hier oder net 3 ')
    # logging.info(f' HÄÄÄÄÄÄÄÄÄ bis hier oder net 3')
    
    



    return f'Selected model is supported by vLLM!', gr.update(visible=True), gr.update(visible=True)

def network_to_pd():       
    rows = []
    try:
        network_list = get_network_data()
        # logging.info(f'[network_to_pd] network_list: {network_list}')  # Use logging.info instead of logging.exception
        for entry in network_list:

            rows.append({
                "container": entry["container"],
                "current_dl": entry["current_dl"]
            })
            
            
        df = pd.DataFrame(rows)
        return df
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        rows.append({
                "container": "0",
                "current_dl": f'0',
                "timestamp": f'0',
                "info": f'0'
        })
        df = pd.DataFrame(rows)
        return df

def container_to_pd():       
    rows = []
    try:
        container_list = get_container_data()
        # print("container_list")
        # print(container_list)
        # logging.info(f'[container_to_pd] container_list: {container_list}')  # Use logging.info instead of logging.exception
        for entry in container_list:
            container_info = ast.literal_eval(entry['container_info'])  # Parse the string into a dictionary
            rows.append({
                "container_i": entry["container_i"]
            })
        df = pd.DataFrame(rows)
        return df
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        rows.append({
                "network_i": "0"
        })
        df = pd.DataFrame(rows)
        return df






def disk_to_pd():
    rows = []
    try:
        disk_list = get_disk_data()
        for entry in disk_list:
            disk_info = ast.literal_eval(entry['disk_info'])
            rows.append({                
                "disk_i": entry.get("disk_i", "0"),
                "timestamp": entry.get("timestamp", "0"),
                "device": disk_info.get("device", "0"),
                "usage_percent": disk_info.get("usage_percent", "0"),
                "mountpoint": disk_info.get("mountpoint", "0"),
                "fstype": disk_info.get("fstype", "0"),
                "opts": disk_info.get("opts", "0"),
                "usage_total": disk_info.get("usage_total", "0"),
                "usage_used": disk_info.get("usage_used", "0"),
                "usage_free": disk_info.get("usage_free", "0"),                
                "io_read_count": disk_info.get("io_read_count", "0"),
                "io_write_count": disk_info.get("io_write_count", "0")                
            })
        df = pd.DataFrame(rows)
        return df
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        logging.info(f' &&&&&& [ERROR] [disk_to_pd] GOT e {e}')

disk_to_pd()

def get_redis(req_db_name, req_container_value):
    res_db_list = r.lrange(req_db_name, 0, -1)
    res_db_list_json = [json.loads(entry) for entry in res_db_list]
    if res_db_list:
        # res_db_list = [json.loads(entry) for entry in res_db_list]
        # print(f'res_db_list: {res_db_list}')
        # res_db_filtered = [entry for entry in res_db_list_json if entry["container"] == req_container_value]
        res_db_filtered_kappa = [entry for entry in res_db_list_json]
        # print(f'res_db_filtered: {res_db_filtered}')
        return res_db_filtered_kappa
    else:
        print(f'No data found for {req_db_name}')
        return []

# all_b = get_redis("db_vllm","b")
# print(f'all_b')
# print(f'{all_b}')

# aaaaa
redis_data = {"db_name": "db_vllm", "vllm_id": "10", "model": "blabla", "ts": "123"}
def vllm_to_pd():
    rows = []
    try:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_to_pd] ** getting vllm_db data ...')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_to_pd] ** getting vllm_db data ...')
        # vllm_list = get_vllm_data()
        vllm_list = get_redis("db_vllm","b")
        
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_to_pd] got vllm_list: {vllm_list}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_to_pd] got vllm_list: {vllm_list}')
        for entry in vllm_list:
            # vllm_info = ast.literal_eval(entry['vllm_info'])
            rows.append({                
                "db_name": entry.get("db_name", "0"),
                "vllm_id": entry.get("vllm_id", "0"),
                "model": entry.get("model", "0"),
                "ts": entry.get("ts", "0")            
            })
        df = pd.DataFrame(rows)
        return df
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        logging.info(f' &&&&&& !!!  [ERROR] [vllm_to_pd] GOT e {e}')

vllm_to_pd()

def gpu_to_pd():
    global GLOBAL_MEM_TOTAL
    global GLOBAL_MEM_USED
    global GLOBAL_MEM_FREE
    rows = []

    try:
        gpu_list = get_gpu_data()
        GLOBAL_MEM_TOTAL = 0
        GLOBAL_MEM_USED = 0
        GLOBAL_MEM_FREE = 0
        for entry in gpu_list:
            gpu_info = ast.literal_eval(entry['gpu_info'])
            
            current_gpu_mem_total = gpu_info.get("mem_total", "0")
            current_gpu_mem_used = gpu_info.get("mem_used", "0")
            current_gpu_mem_free = gpu_info.get("mem_free", "0")
            GLOBAL_MEM_TOTAL = float(GLOBAL_MEM_TOTAL) + float(current_gpu_mem_total.split()[0])
            GLOBAL_MEM_USED = float(GLOBAL_MEM_USED) + float(current_gpu_mem_used.split()[0])
            GLOBAL_MEM_FREE = float(GLOBAL_MEM_FREE) + float(current_gpu_mem_free.split()[0])

            
            rows.append({                                
                "name": gpu_info.get("name", "0"),
                "mem_util": gpu_info.get("mem_util", "0"),
                "timestamp": entry.get("timestamp", "0"),
                "fan_speed": gpu_info.get("fan_speed", "0"),
                "temperature": gpu_info.get("temperature", "0"),
                "gpu_util": gpu_info.get("gpu_util", "0"),
                "power_usage": gpu_info.get("power_usage", "0"),
                "clock_info_graphics": gpu_info.get("clock_info_graphics", "0"),
                "clock_info_mem": gpu_info.get("clock_info_mem", "0"),                
                "cuda_cores": gpu_info.get("cuda_cores", "0"),
                "compute_capability": gpu_info.get("compute_capability", "0"),
                "current_uuid": gpu_info.get("current_uuid", "0"),
                "gpu_i": entry.get("gpu_i", "0"),
                "supported": gpu_info.get("supported", "0"),
                "not_supported": gpu_info.get("not_supported", "0"),
                "status": "ok"
            })

        df = pd.DataFrame(rows)
        return df
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')

gpu_to_pd()




def refresh_container():
    try:
        global docker_container_list
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method": "list"})
        docker_container_list = response.json()
        return docker_container_list
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'err {str(e)}'

            
@dataclass
class VllmCreateComponents:
    method: gr.Textbox
    image: gr.Textbox
    runtime: gr.Textbox
    shm_size: gr.Slider
    port: gr.Slider
    max_model_len: gr.Slider
    tensor_parallel_size: gr.Number
    gpu_memory_utilization: gr.Slider
    
    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]

@dataclass
class VllmCreateValues:
    method: str
    image: str
    runtime: str
    shm_size: int
    port: int
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: int
            
@dataclass
class VllmLoadComponents:
    method: gr.Textbox
    image: gr.Textbox
    runtime: gr.Textbox
    shm_size: gr.Slider
    port: gr.Slider
    max_model_len: gr.Slider
    tensor_parallel_size: gr.Number
    gpu_memory_utilization: gr.Slider
    
    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]

@dataclass
class VllmLoadValues:
    method: str
    image: str
    runtime: str
    shm_size: int
    port: int
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: int



@dataclass
class PromptComponents:
    vllms2: gr.Radio
    port: gr.Slider
    prompt: gr.Textbox
    top_p: gr.Slider
    temperature: gr.Slider
    max_tokens: gr.Slider
    
    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]

@dataclass
class PromptValues:
    vllms2: str
    port: int
    prompt: str
    top_p: int
    temperature: int
    max_tokens: int


BACKEND_URL = f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest'

def docker_api(req_type,req_model=None,req_task=None,req_prompt=None,req_temperature=None, req_config=None):
    
    try:
        print(f'got model_config: {req_config} ')
        response = requests.post(BACKEND_URL, json={
            "req_type":req_type,
            "req_model":req_model,
            "req_task":req_task,
            "req_prompt":req_prompt,
            "req_temperature":req_temperature,
            "req_model_config":req_config
        })
        
        if response.status_code == 200:
            response_json = response.json()
            if response_json["result_status"] != 200:
                logging.exception(f'[docker_api] Response Error: {response_json["result_data"]}')
            return response_json["result_data"]                
        else:
            logging.exception(f'[docker_api] Request Error: {response}')
            return f'Request Error: {response}'
    
    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e


def toggle_vllm_load_create(vllm_list):
    
    if "Create New" in vllm_list:
        return (
            gr.Accordion(open=False,visible=False),
            gr.Button(visible=False),
            gr.Accordion(open=True,visible=True),
            gr.Button(visible=True)
        )

    return (
        gr.Accordion(open=True,visible=True),
        gr.Button(visible=True),    
        gr.Accordion(open=False,visible=False),
        gr.Button(visible=False)
    )

def load_vllm_running3(*params):
    
    try:
        global GLOBAL_SELECTED_MODEL_ID
        print(f' >>> load_vllm_running GLOBAL_SELECTED_MODEL_ID: {GLOBAL_SELECTED_MODEL_ID} ')
        print(f' >>> load_vllm_running got params: {params} ')
        logging.info(f'[load_vllm_running] >> GLOBAL_SELECTED_MODEL_ID: {GLOBAL_SELECTED_MODEL_ID} ')
        logging.info(f'[load_vllm_running] >> got params: {params} ')
                
        req_params = VllmLoadValues(*params)


        response = requests.post(BACKEND_URL, json={
            "req_method":"cleartorch",
            "model_id":GLOBAL_SELECTED_MODEL_ID,
            "max_model_len":req_params.max_model_len,
            "tensor_parallel_size":req_params.tensor_parallel_size,
            "gpu_memory_utilization":req_params.gpu_memory_utilization
        }, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            print(f' !?!?!?!? got response == 200 building json ... {response} ')
            logging.info(f'!?!?!?!? got response == 200 building json ...  {response} ')
            res_json = response.json()        
            print(f' !?!?!?!? GOT RES_JSON: load_vllm_running GLOBAL_SELECTED_MODEL_ID: {res_json} ')
            logging.info(f'!?!?!?!? GOT RES_JSON: {res_json} ')          
            return f'{res_json}'
        else:
            logging.info(f'[load_vllm_running] Request Error: {response}')
            return f'Request Error: {response}'
    
    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'{e}'
    
    
def load_vllm_running2(*params):
    
    try:
        global GLOBAL_SELECTED_MODEL_ID
        print(f' >>> load_vllm_running GLOBAL_SELECTED_MODEL_ID: {GLOBAL_SELECTED_MODEL_ID} ')
        print(f' >>> load_vllm_running got params: {params} ')
        logging.exception(f'[load_vllm_running] >> GLOBAL_SELECTED_MODEL_ID: {GLOBAL_SELECTED_MODEL_ID} ')
        logging.exception(f'[load_vllm_running] >> got params: {params} ')
                
        req_params = VllmLoadValues(*params)


        response = requests.post(BACKEND_URL, json={
            "req_method":"clearsmi",
            "model_id":GLOBAL_SELECTED_MODEL_ID,
            "max_model_len":req_params.max_model_len,
            "tensor_parallel_size":req_params.tensor_parallel_size,
            "gpu_memory_utilization":req_params.gpu_memory_utilization
        }, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            print(f' !?!?!?!? got response == 200 building json ... {response} ')
            logging.exception(f'!?!?!?!? got response == 200 building json ...  {response} ')
            res_json = response.json()        
            print(f' !?!?!?!? GOT RES_JSON: load_vllm_running GLOBAL_SELECTED_MODEL_ID: {res_json} ')
            logging.exception(f'!?!?!?!? GOT RES_JSON: {res_json} ')          
            return f'{res_json}'
        else:
            logging.exception(f'[load_vllm_running] Request Error: {response}')
            return f'Request Error: {response}'
    
    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'{e}'
    
    
    
def llm_load(*params):
    
    try:
        global GLOBAL_SELECTED_MODEL_ID
        print(f' >>> llm_load GLOBAL_SELECTED_MODEL_ID: {GLOBAL_SELECTED_MODEL_ID} ')
        print(f' >>> llm_load got params: {params} ')
        logging.exception(f'[llm_load] >> GLOBAL_SELECTED_MODEL_ID: {GLOBAL_SELECTED_MODEL_ID} ')
        logging.exception(f'[llm_load] >> got params: {params} ')
                
        req_params = VllmLoadComponents(*params)

        response = requests.post(BACKEND_URL, json={
            "req_method":req_params.method,
            "req_image":req_params.image,
            "req_runtime":req_params.runtime,
            "req_shm_size":f'{str(req_params.shm_size)}gb',
            "req_port":req_params.port,
            "req_model":GLOBAL_SELECTED_MODEL_ID,
            "req_tensor_parallel_size":req_params.tensor_parallel_size,
            "req_gpu_memory_utilization":req_params.gpu_memory_utilization,
            "req_max_model_len":req_params.max_model_len
        }, timeout=REQUEST_TIMEOUT)


        if response.status_code == 200:
            print(f' [llm_load] >> got response == 200 building json ... {response} ')
            logging.exception(f'[llm_load] >> got response == 200 building json ...  {response} ')
            res_json = response.json()        
            print(f' [llm_load] >> GOT RES_JSON: GLOBAL_SELECTED_MODEL_ID: {res_json} ')         
            return f'{res_json}'
        else:
            logging.exception(f'[llm_load] Request Error: {response}')
            return f'Request Error: {response}'
    
    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'{e}'
        
    
def llm_create(*params):
    
    try:
        global GLOBAL_SELECTED_MODEL_ID
        print(f' >>> llm_create GLOBAL_SELECTED_MODEL_ID: {GLOBAL_SELECTED_MODEL_ID} ')
        print(f' >>> llm_create got params: {params} ')
        logging.exception(f'[llm_create] >> GLOBAL_SELECTED_MODEL_ID: {GLOBAL_SELECTED_MODEL_ID} ')
        logging.exception(f'[llm_create] >> got params: {params} ')
                
        req_params = VllmCreateComponents(*params)

        response = requests.post(BACKEND_URL, json={
            "req_method":req_params.method,
            "req_image":req_params.image,
            "req_runtime":req_params.runtime,
            "req_shm_size":f'{str(req_params.shm_size)}gb',
            "req_port":req_params.port,
            "req_model":GLOBAL_SELECTED_MODEL_ID,
            "req_tensor_parallel_size":req_params.tensor_parallel_size,
            "req_gpu_memory_utilization":req_params.gpu_memory_utilization,
            "req_max_model_len":req_params.max_model_len
        }, timeout=REQUEST_TIMEOUT)


        if response.status_code == 200:
            print(f' [llm_create] >> got response == 200 building json ... {response} ')
            logging.exception(f'[llm_create] >> got response == 200 building json ...  {response} ')
            res_json = response.json()        
            print(f' [llm_create] >> GOT RES_JSON: GLOBAL_SELECTED_MODEL_ID: {res_json} ')         
            return f'{res_json}'
        else:
            logging.exception(f'[llm_create] Request Error: {response}')
            return f'Request Error: {response}'
    
    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'{e}'
    
        
def llm_prompt(*params):
    
    try:
        global GLOBAL_SELECTED_MODEL_ID
        print(f' >>> llm_prompt GLOBAL_SELECTED_MODEL_ID: {GLOBAL_SELECTED_MODEL_ID} ')
        print(f' >>> llm_prompt got params: {params} ')
        logging.info(f'[llm_prompt] >> GLOBAL_SELECTED_MODEL_ID: {GLOBAL_SELECTED_MODEL_ID} ')
        logging.info(f'[llm_prompt] >> got params: {params} ')

        req_params = PromptComponents(*params)


        DEFAULTS_PROMPT = {
            "vllms2": "container_vllm_xoo",
            "port": 1370,
            "prompt": "Tell a joke",
            "top_p": 0.95,
            "temperature": 0.8,
            "max_tokens": 150
        }

        response = requests.post(BACKEND_URL, json={
            "req_method":"generate",
            "vllms2":getattr(req_params, "vllms2", DEFAULTS_PROMPT["vllms2"]),
            "port":getattr(req_params, "port", DEFAULTS_PROMPT["port"]),
            "prompt": getattr(req_params, "prompt", DEFAULTS_PROMPT["prompt"]),
            "top_p":getattr(req_params, "top_p", DEFAULTS_PROMPT["top_p"]),
            "temperature":getattr(req_params, "temperature", DEFAULTS_PROMPT["temperature"]),
            "max_tokens":getattr(req_params, "max_tokens", DEFAULTS_PROMPT["max_tokens"])
        }, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            print(f' !?!?!?!? [llm_prompt] got response == 200 building json ... {response} ')
            logging.info(f'!?!?!?!? [llm_prompt] got response == 200 building json ...  {response} ')
            res_json = response.json()        
            print(f' !?!?!?!? [llm_prompt] GOT RES_JSON: llm_prompt GLOBAL_SELECTED_MODEL_ID: {res_json} ')
            logging.info(f'!?!?!?!? [llm_prompt] GOT RES_JSON: {res_json} ')          
            return f'{res_json}'
        else:
            logging.exception(f'[llm_prompt] Request Error: {response}')
            return f'Request Error: {response}'
    
    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'{e}'
    



def download_from_hf_hub(selected_model_id):
    try:
        selected_model_id_arr = str(selected_model_id).split('/')
        print(f'selected_model_id_arr {selected_model_id_arr}...')       
        model_path = snapshot_download(
            repo_id=selected_model_id,
            local_dir=f'/models/{selected_model_id_arr[0]}/{selected_model_id_arr[1]}'
        )
        return f'Saved to {model_path}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'download error: {e}'


download_info_prev_bytes_recv = 0   
download_info_current_model_bytes_recv = 0    
 
def download_info(req_model_size, progress=gr.Progress()):
    global download_info_prev_bytes_recv
    global download_info_current_model_bytes_recv
    download_info_prev_bytes_recv = 0
    download_info_current_model_bytes_recv = 0
    progress(0, desc="Initializing ...")
    progress(0.01, desc="Calculating Download Time ...")
    
    avg_dl_speed_val = 0
    avg_dl_speed = []
    for i in range(0,5):
        net_io = psutil.net_io_counters()
        bytes_recv = net_io.bytes_recv
        download_speed = bytes_recv - download_info_prev_bytes_recv
        download_speed_mbit_s = (download_speed * 8) / (1024 ** 2) 
        
        download_info_prev_bytes_recv = int(bytes_recv)
        download_info_current_model_bytes_recv = download_info_current_model_bytes_recv + download_info_prev_bytes_recv
        avg_dl_speed.append(download_speed)
        avg_dl_speed_val = sum(avg_dl_speed)/len(avg_dl_speed)
        logging.info(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')
        print(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')  
        time.sleep(1)
    
    logging.info(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')
    print(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')  



    calc_mean = lambda data: np.mean([x for x in data if (np.percentile(data, 25) - 1.5 * (np.percentile(data, 75) - np.percentile(data, 25))) <= x <= (np.percentile(data, 75) + 1.5 * (np.percentile(data, 75) - np.percentile(data, 25)))]) if data else 0


    avg_dl_speed_val = calc_mean(avg_dl_speed)
        
    
    logging.info(f' **************** [download_info] avg_dl_speed_val: {avg_dl_speed_val}')
    print(f' **************** [download_info] avg_dl_speed_val: {avg_dl_speed_val}')    

    est_download_time_sec = int(req_model_size)/int(avg_dl_speed_val)
    logging.info(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')
    print(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')

    est_download_time_sec = int(est_download_time_sec)
    logging.info(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')
    print(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')

    logging.info(f' **************** [download_info] zzz waiting for download_complete_event zzz waiting {est_download_time_sec}')
    print(f' **************** [download_info] zzz waiting for download_complete_event zzz waiting {est_download_time_sec}')
    current_dl_arr = []
    for i in range(0,est_download_time_sec):
        if len(current_dl_arr) > 5:
            current_dl_arr = []
        net_io = psutil.net_io_counters()
        bytes_recv = net_io.bytes_recv
        download_speed = bytes_recv - download_info_prev_bytes_recv
        current_dl_arr.append(download_speed)
        print(f' &&&&&&&&&&&&&& current_dl_arr: {current_dl_arr}')
        if all(value < 10000 for value in current_dl_arr[-4:]):
            print(f' &&&&&&&&&&&&&& DOWNLOAD FINISH EHH??: {current_dl_arr}')
            yield f'Progress: 100%\nFiniiiiiiiish!'
            return
            
        download_speed_mbit_s = (download_speed * 8) / (1024 ** 2)
        
        download_info_prev_bytes_recv = int(bytes_recv)
        download_info_current_model_bytes_recv = download_info_current_model_bytes_recv + download_info_prev_bytes_recv

        progress_percent = (i + 1) / est_download_time_sec
        progress(progress_percent, desc=f"Downloading ... {download_speed_mbit_s:.2f} MBit/s")

        time.sleep(1)
    logging.info(f' **************** [download_info] LOOP DONE!')
    print(f' **************** [download_info] LOOP DONE!')
    yield f'Progress: 100%\nFiniiiiiiiish!'


def parallel_download(selected_model_size, model_dropdown):
    # Create threads for both functions
    thread_info = threading.Thread(target=download_info, args=(selected_model_size,))
    thread_hub = threading.Thread(target=download_from_hf_hub, args=(model_dropdown,))

    # Start both threads
    thread_info.start()
    thread_hub.start()

    # Wait for both threads to finish
    thread_info.join()
    thread_hub.join()

    return "Download finished!"


def create_app():
    with gr.Blocks() as app:
        gr.Markdown(
            """
            # Welcome!
            Select a _[Hugging Face Model](https://huggingface.co/models)_ and deploy it with vLLM
            
            **Note**: _[vLLM supported models list](https://docs.vllm.ai/en/latest/models/supported_models.html)_
            or """)
        btn_tested = gr.Button("select a tested model", size="sm")

        input_search = gr.Textbox(placeholder="Type in a Hugging Face model or tag", show_label=False, autofocus=True)
        btn_search = gr.Button("Search")

        with gr.Row(visible=False) as row_model_select:
            model_dropdown = gr.Dropdown(choices=[''], interactive=True, show_label=False)
        with gr.Row(visible=False) as row_model_info:
            with gr.Column(scale=4):
                with gr.Accordion(("Model Parameters"), open=False):                    
                    with gr.Row():
                        selected_model_id = gr.Textbox(label="id")
                        selected_model_container_name = gr.Textbox(label="container_name")
                        
                        
                    with gr.Row():
                        selected_model_architectures = gr.Textbox(label="architectures")
                        selected_model_pipeline_tag = gr.Textbox(label="pipeline_tag")
                        selected_model_transformers = gr.Textbox(label="transformers")
                        
                        
                    with gr.Row():
                        selected_model_model_type = gr.Textbox(label="model_type")
                        selected_model_quantization = gr.Textbox(label="quantization")
                        selected_model_size = gr.Textbox(label="size")
                        selected_model_torch_dtype = gr.Textbox(label="torch_dtype")        
                        selected_model_hidden_size = gr.Textbox(label="hidden_size")                        
                        
                    with gr.Row():
                        selected_model_private = gr.Textbox(label="private")
                        selected_model_gated = gr.Textbox(label="gated")
                        selected_model_downloads = gr.Textbox(label="downloads")
                                          
                        
                        
                    
                    with gr.Accordion(("Model Configs"), open=False):
                        with gr.Row():
                            selected_model_search_data = gr.Textbox(label="search_data", lines=20, elem_classes="table-cell")
                        with gr.Row():
                            selected_model_hf_data = gr.Textbox(label="hf_data", lines=20, elem_classes="table-cell")
                        with gr.Row():
                            selected_model_config_data = gr.Textbox(label="config_data", lines=20, elem_classes="table-cell")

                    with gr.Row():
                        port_model = gr.Number(value=8001,visible=False,label="Port of model: ")
                        port_vllm = gr.Number(value=8000,visible=False,label="Port of vLLM: ")
                        
        output = gr.Textbox(label="Output", show_label=True, visible=True)   
        # aaaa
        with gr.Row(visible=True) as row_vllm:
            with gr.Column(scale=4):
                
                
                with gr.Row(visible=False) as row_select_vllm:
                    vllms=gr.Radio(["vLLM1", "vLLM2", "Create New"], value="vLLM1", show_label=False, info="Select a vLLM or create a new one. Where?")
                    
                with gr.Accordion(("Create vLLM Parameters"), open=False, visible=True) as acc_create:
                    vllm_create_components = VllmCreateComponents(

                        method=gr.Textbox(value="create", label="method", info=f"yee the req_method."),
                        
                        image=gr.Textbox(value="xoo4foo/zzvllm44:latest", label="image", info=f"Dockerhub vLLM image"),
                        runtime=gr.Textbox(value="nvidia", label="runtime", info=f"Container runtime"),
                        shm_size=gr.Slider(1, 320, step=1, value=8, label="shm_size", info=f'Maximal GPU Memory in GB'),
                        
                        port=gr.Slider(1370, 1380, step=1, value=1375, label="port", info=f"Choose a port."),                        
                        
                        max_model_len=gr.Slider(1024, 8192, step=1024, value=1024, label="max_model_len", info=f"Model context length. If unspecified, will be automatically derived from the model config."),
                        tensor_parallel_size=gr.Number(1, 8, value=1, label="tensor_parallel_size", info=f"Number of tensor parallel replicas."),
                        gpu_memory_utilization=gr.Slider(0.2, 0.99, value=0.87, label="gpu_memory_utilization", info=f"The fraction of GPU memory to be used for the model executor, which can range from 0 to 1.")
                    )
                    
                    

                                        
                with gr.Accordion(("Load vLLM Parameters"), open=False, visible=True) as acc_load:
                    vllm_load_components = VllmLoadComponents(

                        method=gr.Textbox(value="load", label="method", info=f"yee the req_method."),
                        
                        image=gr.Textbox(value="xoo4foo/zzvllm44:latest", label="image", info=f"Dockerhub vLLM image"),
                        runtime=gr.Textbox(value="nvidia", label="runtime", info=f"Container runtime"),
                        shm_size=gr.Slider(1, 320, step=1, value=8, label="shm_size", info=f'Maximal GPU Memory in GB'),
                        
                        port=gr.Slider(1370, 1380, step=1, value=1375, label="port", info=f"Choose a port."),
                                                                        
                        max_model_len=gr.Slider(1024, 8192, step=1024, value=1024, label="max_model_len", info=f"Model context length. If unspecified, will be automatically derived from the model config."),
                        tensor_parallel_size=gr.Number(1, 8, value=1, label="tensor_parallel_size", info=f"Number of tensor parallel replicas."),
                        gpu_memory_utilization=gr.Slider(0.2, 0.99, value=0.87, label="gpu_memory_utilization", info=f"The fraction of GPU memory to be used for the model executor, which can range from 0 to 1.")
                    )
                    
                    



            with gr.Column(scale=1):
                with gr.Row(visible=False) as row_download:
                    btn_dl = gr.Button("DOWNLOAD", variant="primary")
                with gr.Row(visible=False) as vllm_load_actions:
                    btn_load = gr.Button("DEPLOY")
                    # btn_vllm_running2 = gr.Button("CLEAR NU GO 1370")
                    # btn_vllm_running3 = gr.Button("CLEAR TORCH", visible=True)
                with gr.Row(visible=False) as vllm_create_actions:
                    btn_create = gr.Button("CREATE", variant="primary")
                    btn_create_close = gr.Button("CANCEL")
            
        with gr.Accordion(("Prompt Parameters"), open=False, visible=True) as acc_prompt:
            with gr.Column(scale=2):
                llm_prompt_components = PromptComponents(
                    vllms2=gr.Radio(["vLLM xoo (1370)", "vLLM oai (1371)", "Create New"], value="vLLM_xoo_1370", show_label=False, info="Select a vllms_prompt or create a new one. Where?"),
                    port=gr.Slider(1370, 1380, step=1, value=1375, label="port", info=f"Choose a port."),
                    prompt = gr.Textbox(placeholder="Ask a question", value="Follow the", label="Prompt", show_label=True, visible=True),
                    top_p=gr.Slider(0.01, 1.0, step=0.01, value=0.95, label="top_p", info=f'Float that controls the cumulative probability of the top tokens to consider'),
                    temperature=gr.Slider(0.0, 0.99, step=0.01, value=0.8, label="temperature", info=f'Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling'),
                    max_tokens=gr.Slider(50, 2500, step=25, value=150, label="max_tokens", info=f'Maximum number of tokens to generate per output sequence')
                )  
            with gr.Column(scale=1):
                with gr.Row() as vllm_prompt_output:
                    output_prompt = gr.Textbox(label="Prompt Output", show_label=True)
                with gr.Row() as vllm_prompt:
                    prompt_btn = gr.Button("PROMPT")


        with gr.Accordion(("System Stats"), open=False) as acc_system_stats:
            
            with gr.Accordion(("GPU information"), open=False) as acc_gpu_dataframe:
                gpu_dataframe = gr.Dataframe()

            with gr.Accordion(("vLLM information"), open=False) as acc_vllm_dataframe:
                vllm_dataframe = gr.Dataframe()

            with gr.Accordion(("Disk information"), open=False) as acc_disk_dataframe:
                disk_dataframe = gr.Dataframe()

            with gr.Accordion(("Network information"), open=False) as acc_network_dataframe:
                network_dataframe = gr.Dataframe()


        vllm_timer = gr.Timer(1,active=True)
        vllm_timer.tick(vllm_to_pd, outputs=vllm_dataframe)
        
        disk_timer = gr.Timer(1,active=True)
        disk_timer.tick(disk_to_pd, outputs=disk_dataframe)

        gpu_timer = gr.Timer(1,active=True)
        gpu_timer.tick(gpu_to_pd, outputs=gpu_dataframe)

        network_timer = gr.Timer(1,active=True)
        network_timer.tick(network_to_pd, outputs=network_dataframe)







        with gr.Column(scale=1, visible=True) as vllm_running_engine_argumnts_btn:
            vllm_running_engine_arguments_show = gr.Button("LOAD VLLM CREATEEEEEEEEUUUUHHHHHHHH", variant="primary")
            vllm_running_engine_arguments_close = gr.Button("CANCEL")

                

    

                        
        with gr.Row(visible=True) as row_audio:
            gr.Markdown("## Faster-Whisper Audio Transcription")
            gr.Markdown("Upload an audio file to transcribe it using a faster-whisper model.")
            with gr.Column(scale=2):
                audio_input = gr.Audio(label="Upload Audio", type="filepath")
                

            with gr.Column(scale=1):
                text_output = gr.Textbox(label="Transcription", lines=10)
            
            transcribe_btn = gr.Button("Transcribe")
            transcribe_btn.click(
                transcribe_audio,
                inputs=audio_input,
                outputs=text_output
            )
        
                

            
        btn_interface = gr.Button("Load Interface",visible=False)
        @gr.render(inputs=[selected_model_pipeline_tag, selected_model_id], triggers=[btn_interface.click])
        def show_split(text_pipeline, text_model):
            if len(text_model) == 0:
                gr.Markdown("Error pipeline_tag or model_id")
            else:
                selected_model_id_arr = str(text_model).split('/')
                print(f'selected_model_id_arr {selected_model_id_arr}...')            
                gr.Interface.from_pipeline(pipeline(text_pipeline, model=f'/models/{selected_model_id_arr[0]}/{selected_model_id_arr[1]}'))

        timer_c = gr.Timer(1,active=False)
        timer_c.tick(refresh_container)
                










        
        
        
        
        
        
        
        
        
        

        
        
        
       
        load_btn = gr.Button("Load into vLLM (port: 1370)", visible=True)

        

        
        container_state = gr.State([])   
        docker_container_list = get_docker_container_list()     
        @gr.render(inputs=container_state)
        def render_container(render_container_list):
            docker_container_list = get_docker_container_list()
            docker_container_list_sys_running = [c for c in docker_container_list if c["State"]["Status"] == "running" and c["Name"] in ["/container_redis","/container_backend", "/container_frontend"]]
            docker_container_list_sys_not_running = [c for c in docker_container_list if c["State"]["Status"] != "running" and c["Name"] in ["/container_redis","/container_backend", "/container_frontend"]]
            docker_container_list_vllm_running = [c for c in docker_container_list if c["State"]["Status"] == "running" and c["Name"] not in ["/container_redis","/container_backend", "/container_frontend"]]
            docker_container_list_vllm_not_running = [c for c in docker_container_list if c["State"]["Status"] != "running" and c["Name"] not in ["/container_redis","/container_backend", "/container_frontend"]]

            def refresh_container():
                try:
                    global docker_container_list
                    response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method": "list"})
                    docker_container_list = response.json()
                    return docker_container_list
                
                except Exception as e:
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                    return f'err {str(e)}'


            with gr.Accordion(f'vLLM | Running {len(docker_container_list_vllm_running)} | Not Running {len(docker_container_list_vllm_not_running)}', open=False):
                gr.Markdown(f'### Running ({len(docker_container_list_vllm_running)})')

                for current_container in docker_container_list_vllm_running:
                    with gr.Row():
                        
                        container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                        
                        container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
            
                        container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                        
                        container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                        
                    with gr.Row():
                        container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)

                    with gr.Row():
                        btn_logs_docker_open = gr.Button("Docker Log", scale=0)
                        btn_logs_docker_close = gr.Button("Close Docker Log", scale=0, visible=False)     
                        
                        btn_logs_docker_open.click(
                            docker_api_logs,
                            [container_id],
                            [container_log_out]
                        ).then(
                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                        
                        btn_logs_docker_close.click(
                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                        
                        stop_btn = gr.Button("Stop", scale=0)
                        delete_btn = gr.Button("Delete", scale=0, variant="stop")

                        stop_btn.click(
                            docker_api_stop,
                            [container_id],
                            [container_state]
                        ).then(
                            refresh_container,
                            outputs=[container_state]
                        )

                        delete_btn.click(
                            docker_api_delete,
                            [container_id],
                            [container_state]
                        ).then(
                            refresh_container,
                            outputs=[container_state]
                        )
                        
                    gr.Markdown(
                        """
                        <hr>
                        """
                    )


                gr.Markdown(f'### Not running ({len(docker_container_list_vllm_not_running)})')

                for current_container in docker_container_list_vllm_not_running:
                    with gr.Row():
                        
                        container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container ID")
                        
                        container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
            
                        container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                        
                        container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                    
                    with gr.Row():
                        container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)
                        
                    with gr.Row():
                        btn_logs_docker_open = gr.Button("Docker Log", scale=0)
                        btn_logs_docker_close = gr.Button("Close Docker Log", scale=0, visible=False)     
                        
                        btn_logs_docker_open.click(
                            docker_api_logs,
                            [container_id],
                            [container_log_out]
                        ).then(
                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                        
                        btn_logs_docker_close.click(
                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                        
                        start_btn = gr.Button("Start", scale=0)
                        delete_btn = gr.Button("Delete", scale=0, variant="stop")

                        start_btn.click(
                            docker_api_start,
                            [container_id],
                            [container_state]
                        ).then(
                            refresh_container,
                            [container_state]
                        )

                        delete_btn.click(
                            docker_api_delete,
                            [container_id],
                            [container_state]
                        ).then(
                            refresh_container,
                            outputs=[container_state]
                        )
                    
                    gr.Markdown(
                        """
                        <hr>
                        """
                    )
                    
            

            with gr.Accordion(f'System | Running {len(docker_container_list_sys_running)} | Not Running {len(docker_container_list_sys_not_running)}', open=False):
                gr.Markdown(f'### Running ({len(docker_container_list_sys_running)})')

                for current_container in docker_container_list_sys_running:
                    with gr.Row():
                        
                        container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                        
                        container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
            
                        container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                        
                        container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                        
                    with gr.Row():
                        container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)

                    with gr.Row():            
                        btn_logs_file_open = gr.Button("Log File", scale=0)
                        btn_logs_file_close = gr.Button("Close Log File", scale=0, visible=False)   
                        btn_logs_docker_open = gr.Button("Docker Log", scale=0)
                        btn_logs_docker_close = gr.Button("Close Docker Log", scale=0, visible=False)     

                        btn_logs_file_open.click(
                            load_log_file,
                            [container_name],
                            [container_log_out]
                        ).then(
                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                        )
                        
                        btn_logs_file_close.click(
                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                        )
                        
                        btn_logs_docker_open.click(
                            docker_api_logs,
                            [container_id],
                            [container_log_out]
                        ).then(
                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                        
                        btn_logs_docker_close.click(
                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                        
                    gr.Markdown(
                        """
                        <hr>
                        """
                    )

                gr.Markdown(f'### Not Running ({len(docker_container_list_sys_not_running)})')

                for current_container in docker_container_list_sys_not_running:
                    with gr.Row():
                        
                        container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container ID")
                        
                        container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
            
                        container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                        
                        container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                    
                    with gr.Row():
                        container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)
                        
                    with gr.Row():
                        btn_logs_file_open = gr.Button("Log File", scale=0)
                        btn_logs_file_close = gr.Button("Close Log File", scale=0, visible=False)   
                        btn_logs_docker_open = gr.Button("Docker Log", scale=0)
                        btn_logs_docker_close = gr.Button("Close Docker Log", scale=0, visible=False)     

                        btn_logs_file_open.click(
                            load_log_file,
                            [container_name],
                            [container_log_out]
                        ).then(
                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                        )
                        
                        btn_logs_file_close.click(
                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                        )
                        
                        btn_logs_docker_open.click(
                            docker_api_logs,
                            [container_id],
                            [container_log_out]
                        ).then(
                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                        
                        btn_logs_docker_close.click(
                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                        )
                    
                    gr.Markdown(
                        """
                        <hr>
                        """
                    )











            
        
        
        
        def refresh_container_list():
            try:
                global docker_container_list
                response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method": "list"})
                docker_container_list = response.json()
                return docker_container_list
            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return f'err {str(e)}'



        input_search.submit(
            search_models, 
            input_search, 
            [model_dropdown]
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            model_dropdown
        )
        
        btn_search.click(
            search_models, 
            input_search, 
            [model_dropdown]
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            model_dropdown
        )


        btn_dl.click(
            parallel_download, 
            [selected_model_size, model_dropdown], 
            output,
            concurrency_limit=15
        )


        btn_dl.click(
            lambda: gr.update(label="Starting download",visible=True),
            None,
            output
        ).then(
            download_info, 
            selected_model_size,
            output,
            concurrency_limit=15
        ).then(
            download_from_hf_hub, 
            model_dropdown,
            output,
            concurrency_limit=15
        ).then(
            lambda: gr.update(label="Download finished"),
            None,
            output
        ).then(
            lambda: gr.update(visible=False),
            None,
            row_download
        ).then(
            lambda: gr.update(visible=True),
            None,
            btn_interface
        ).then(
            lambda: gr.update(visible=True),
            None,
            acc_load
        ).then(
            lambda: gr.update(visible=True),
            None,
            vllm_load_actions
        ).then(
            lambda: gr.update(visible=True),
            None,
            row_select_vllm
        ).then(
            lambda: gr.update(visible=True, open=False),
            None,
            acc_load
        )






        input_search.submit(
            search_models, 
            input_search, 
            [model_dropdown]
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            model_dropdown
        )
        
        btn_search.click(
            search_models, 
            input_search, 
            [model_dropdown]
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            model_dropdown
        )




        model_dropdown.change(
            get_info, 
            model_dropdown, 
            [selected_model_search_data,selected_model_id,selected_model_architectures,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_downloads,selected_model_container_name]
        ).then(
            get_additional_info, 
            model_dropdown, 
            [selected_model_hf_data, selected_model_config_data, selected_model_architectures,selected_model_id, selected_model_size, selected_model_gated, selected_model_model_type, selected_model_quantization, selected_model_torch_dtype, selected_model_hidden_size]
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            row_model_select
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            row_model_info
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            row_vllm
        ).then(
            gr_load_check, 
            [selected_model_id, selected_model_architectures, selected_model_pipeline_tag, selected_model_transformers, selected_model_size, selected_model_private, selected_model_gated, selected_model_model_type, selected_model_quantization],
            [output,row_download,btn_load]
        )


        vllm_running_engine_arguments_show.click(
            lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], 
            None, 
            [vllm_running_engine_arguments_show, vllm_running_engine_arguments_close, acc_load]
        )
        
        vllm_running_engine_arguments_close.click(
            lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], 
            None, 
            [vllm_running_engine_arguments_show, vllm_running_engine_arguments_close, acc_load]
        )




        btn_load.click(
            lambda: gr.update(label="Deploying"),
            None,
            output
        ).then(
            lambda: gr.update(visible=True, open=False), # hier
            None, 
            acc_load    
        ).then(
            lambda: gr.update(visible=True), # hier
            None, 
            row_select_vllm   
        ).then(
            llm_load,
            vllm_load_components.to_list(),
            [output]
        ).then(
            lambda: gr.update(visible=True, open=True), 
            None, 
            acc_prompt
        ).then(
            lambda: gr.update(visible=True), # hier
            None, 
            btn_load
        ).then(
            lambda: gr.update(visible=True, open=True), 
            None, 
            acc_prompt
        ).then(
            refresh_container,
            [container_state]
        )

        btn_create.click(
            lambda: gr.update(label="Deploying"),
            None,
            output
        ).then(
            lambda: gr.update(visible=True, open=False), 
            None, 
            acc_create    
        ).then(
            llm_create,
            vllm_create_components.to_list(),
            [output]
        ).then(
            lambda: gr.update(visible=True, open=True),
            None, 
            acc_prompt
        ).then(
            lambda: gr.update(visible=True), # hier
            None, 
            btn_create
        ).then(
            lambda: gr.update(visible=True), # hier
            None, 
            btn_create_close
        ).then(
            refresh_container,
            [container_state]
        )


        
        prompt_btn.click(
            llm_prompt,
            llm_prompt_components.to_list(),
            [output_prompt]
        )


        vllms.change(
            toggle_vllm_load_create,
            vllms,
            [acc_load, vllm_load_actions, acc_create, vllm_create_actions]
        )

        vllms.change(
            toggle_vllm_load_create,
            vllms,
            [acc_load, vllm_load_actions, acc_create, vllm_create_actions]
        )




    return app

# Launch the app
if __name__ == "__main__":
    backend_url = f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest'
    
    # Wait for the backend container to be online
    if wait_for_backend(backend_url):
        app = create_app()
        app.launch(server_name=f'{os.getenv("FRONTEND_IP")}', server_port=int(os.getenv("FRONTEND_PORT")))
    else:
        print("Failed to start application due to backend container not being online.")
