from dataclasses import dataclass, fields
import gradio as gr
import redis
import time
import os
import json
import huggingface_hub
from huggingface_hub import snapshot_download
import logging
from datetime import datetime
import requests
from requests.exceptions import Timeout
import huggingface_hub
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import git
from git import Repo


import subprocess

current_models_data = []
db_gpu_data = []
db_gpu_data_len = ''

GLOBAL_SEARCH_INPUT_TS = 0
GLOBAL_SEARCH_INPUT_THRESHOLD = 10
REQUEST_TIMEOUT = 3
GLOBAL_SEARCH_INITIAL_DELAY = 10



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
    logging.info(f' [START] {len(defaults_frontend['tested_models'])} tested_models found!')
    logging.info(f' [START] {len(defaults_frontend['audio_models'])} audio_models found!')











model_dropdown = gr.Dropdown(choices=[''], label=f'Select a Hugging Face model', interactive=True, show_label=False, visible=False)

        
        
def dropdown_load_tested_models():
    global current_models_data
    print(f'len(defaults_frontend["tested_models"]): {len(defaults_frontend["tested_models"])}')
    current_models_data = defaults_frontend["tested_models"].copy()
    model_ids = [m["id"] for m in defaults_frontend["tested_models"]]
    print(f'model_ids: {model_ids}')
    # return gr.update(choices=model_ids, value=response_models[0]["id"], visible=True)
    return [gr.update(choices=model_ids, value=defaults_frontend["tested_models"][0]["id"], visible=True),gr.update(show_label=True, label=f'Loaded {len(model_ids)} models!')]
    # return [gr.update(choices=model_ids, value=defaults_frontend["tested_models"][0]["id"], visible=True),gr.update(value=defaults_frontend["tested_models"][0]["id"],show_label=True, label=f'Loaded {len(model_ids)} models!')]

def huggingface_hub_search(query):
    try:
        global current_models_data
        response = requests.get(f'https://huggingface.co/api/models?search={query}', timeout=REQUEST_TIMEOUT)
        response_models = response.json()
        print(f'response_models: {response_models}')
        current_models_data = response_models.copy()
        model_ids = [m["id"] for m in response_models]
        print(f'model_ids: {model_ids}')
        if len(response_models) >= 1000:
            return [model_ids, response_models[0]["id"]]
        return [model_ids, response_models[0]["id"]]
    except Timeout:
        return [[''], f'Timeout for {response}']

    except Exception as e:
        return [[''], f'{str(e)}']



def search_change(input_text):
    global GLOBAL_SEARCH_INPUT_TS
    global current_models_data
    current_ts = int(datetime.now().timestamp())
    if GLOBAL_SEARCH_INPUT_TS + GLOBAL_SEARCH_INPUT_THRESHOLD > current_ts:
        wait_time = GLOBAL_SEARCH_INPUT_TS + GLOBAL_SEARCH_INPUT_THRESHOLD - current_ts
        return [gr.update(show_label=False),gr.update(show_label=True, label=f'Found {len(current_models_data)} models! Please wait {wait_time} sec or click on search')]
    if len(input_text) < 3: 
        # return [gr.update(show_label=False),gr.update(show_label=True, label=" < 3")]
        return [gr.update(show_label=False),gr.update(show_label=True)]
    if GLOBAL_SEARCH_INPUT_TS == 0 and len(input_text) > 5:
        GLOBAL_SEARCH_INPUT_TS = int(datetime.now().timestamp())
        res_huggingface_hub_search_model_ids,  res_huggingface_hub_search_current_value = huggingface_hub_search(input_text)
        if len(res_huggingface_hub_search_model_ids) >= 1000:
            return [gr.update(choices=res_huggingface_hub_search_model_ids, value=res_huggingface_hub_search_current_value, visible=True),gr.update(show_label=True, label=f'Found >1000 models!')]
        return [gr.update(choices=res_huggingface_hub_search_model_ids, value=res_huggingface_hub_search_current_value, visible=True),gr.update(show_label=True, label=f'Found {len(res_huggingface_hub_search_model_ids)} models!')]
        
    if GLOBAL_SEARCH_INPUT_TS == 0:
        GLOBAL_SEARCH_INPUT_TS = int(datetime.now().timestamp()) + GLOBAL_SEARCH_INITIAL_DELAY
        return [gr.update(show_label=False),gr.update(show_label=True, label=f'Waiting auto search {GLOBAL_SEARCH_INITIAL_DELAY} sec')]
    if GLOBAL_SEARCH_INPUT_TS + GLOBAL_SEARCH_INPUT_THRESHOLD <= current_ts:
        GLOBAL_SEARCH_INPUT_TS = int(datetime.now().timestamp())
        res_huggingface_hub_search_model_ids,  res_huggingface_hub_search_current_value = huggingface_hub_search(input_text)
        if len(res_huggingface_hub_search_model_ids) >= 1000:
            return [gr.update(choices=res_huggingface_hub_search_model_ids, value=res_huggingface_hub_search_current_value, visible=True),gr.update(show_label=True, label=f'Found >1000 models!')]
        return [gr.update(choices=res_huggingface_hub_search_model_ids, value=res_huggingface_hub_search_current_value, visible=True),gr.update(show_label=True, label=f'Found {len(res_huggingface_hub_search_model_ids)} models!')]



def search_models(query):
    try:
        
        global current_models_data    
        response = requests.get(f'https://huggingface.co/api/models?search={query}', timeout=REQUEST_TIMEOUT)
        response_models = response.json()
        current_models_data = response_models.copy()
        model_ids = [m["id"] for m in response_models]
        if len(response_models) >= 1000:
            return [gr.update(choices=model_ids, value=response_models[0]["id"]),gr.update(label=f'found > {len(response_models)} models!')]
        return [gr.update(choices=model_ids, value=response_models[0]["id"]),gr.update(label=f'found {len(response_models)} models!')]
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')




def format_bytes(req_format, req_size):
    if req_format == "human":
        for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
            if abs(req_size) < 1024.0:
                return f'{req_size:3.1f}{unit}B'
            req_size /= 1024.0
        return f'{req_size:.1f}YiB'
    elif req_format == "bytes":
        req_size = req_size.upper()
        if 'KB' in req_size:
            return int(float(req_size.replace('KB', '').strip()) * 1024)
        elif 'MB' in req_size:
            return int(float(req_size.replace('MB', '').strip()) * 1024 * 1024)
        elif 'GB' in req_size:
            return int(float(req_size.replace('GB', '').strip()) * 1024 * 1024 * 1024)
        elif 'B' in req_size:
            return int(float(req_size.replace('B', '').strip()))
        return 0
    else:
        raise ValueError("Invalid format specified. Use 'human' or 'bytes'.")


def get_git_model_size(selected_id):    
    try:
        repo = Repo.clone_from(f'https://huggingface.co/{selected_id}', selected_id, no_checkout=True)
    except git.exc.GitCommandError as e:
        if "already exists and is not an empty directory" in str(e):
            repo = Repo(selected_id)
        else:
            raise
    
    lfs_files = repo.git.lfs("ls-files", "-s").splitlines()
    files_list = []
    for line in lfs_files:
        parts = line.split(" - ")
        if len(parts) == 2:
            file_hash, file_info = parts
            file_parts = file_info.rsplit(" (", 1)
            if len(file_parts) == 2:
                file_name = file_parts[0]
                size_str = file_parts[1].replace(")", "")
                size_bytes = format_bytes("bytes",size_str)
                
                files_list.append({
                    "id": file_hash.strip(),
                    "file": file_name.strip(),
                    "size": size_bytes,
                    "size_human": size_str
                })
            
        
    return sum([file["size"] for file in files_list]), format_bytes("human",sum([file["size"] for file in files_list]))
    


def convert_to_bytes(size_str):
    """Convert human-readable file size to bytes"""
    size_str = size_str.upper()
    if 'KB' in size_str:
        return int(float(size_str.replace('KB', '').strip()) * 1024)
    elif 'MB' in size_str:
        return int(float(size_str.replace('MB', '').strip()) * 1024 * 1024)
    elif 'GB' in size_str:
        return int(float(size_str.replace('GB', '').strip()) * 1024 * 1024 * 1024)
    elif 'B' in size_str:
        return int(float(size_str.replace('B', '').strip()))
    return 0


# def get_git_model_size(selected_id):    
#   repo_url = "https://huggingface.co/intfloat/multilingual-e5-large"
#   repo_name = "multilingual-e5-large"

#   # Clone the repository without checking out files (if not already present)
#   try:
#       subprocess.run(["git", "clone", "--no-checkout", repo_url], check=True)
#   except subprocess.CalledProcessError:
#       # Repository might already exist, continue
#       pass

#   # Get the Git LFS file list
#   result = subprocess.run(
#       ["git", "lfs", "ls-files", "-s"], 
#       cwd=repo_name, 
#       capture_output=True, 
#       text=True, 
#       check=True
#   )
#   print(f'get_git_model_size {get_git_model_size}')
#   # Parse output into the desired format
#   files_list = []
#   for line in result.stdout.splitlines():
#       parts = line.split(" - ")
#       if len(parts) == 2:
#           file_hash, file_info = parts
#           # Handle cases where filename might contain spaces
#           file_parts = file_info.rsplit(" (", 1)
#           if len(file_parts) == 2:
#               file_name = file_parts[0]
#               size_str = file_parts[1].replace(")", "")
#               size_bytes = convert_to_bytes(size_str)
              
#               files_list.append({
#                   "id": file_hash.strip(),
#                   "file": file_name.strip(),
#                   "size": size_bytes,
#                   "size_human": size_str  # Keeping human-readable format as well
#               })

              
        
#   return 0,0
#   return sum([file["size"] for file in files_list]), format_bytes("human",sum([file["size"] for file in files_list]))
    




def get_info(selected_id):
    
    print(f' @@@ [get_info] 0')
    print(f' @@@ [get_info] 0')   
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
        print(f' @@@ [get_info] selected_id NOT FOUND!! RETURN ') 
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
    
    global current_models_data
    global GLOBAL_SELECTED_MODEL_ID
    GLOBAL_SELECTED_MODEL_ID = selected_id
    print(f' @@@ [get_info] {selected_id} 2')
    print(f' @@@ [get_info] {selected_id} 2')  
    
    print(f' @@@ [get_info] {selected_id} 3')
    print(f' @@@ [get_info] {selected_id} 3')  
    container_name = str(res_model_data["model_id"]).replace('/', '_')
    print(f' @@@ [get_info] {selected_id} 4')
    print(f' @@@ [get_info] {selected_id} 4')  
    if len(current_models_data) < 1:
        print(f' @@@ [get_info] len(current_models_data) < 1! RETURN ')
        print(f' @@@ [get_info] len(current_models_data) < 1! RETURN ') 
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
    try:
        print(f' @@@ [get_info] {selected_id} 5')
        print(f' @@@ [get_info] {selected_id} 5') 
        for item in current_models_data:
            print(f' @@@ [get_info] {selected_id} 6')
            print(f' @@@ [get_info] {selected_id} 6') 
            if item['id'] == selected_id:
                print(f' @@@ [get_info] {selected_id} 7')
                print(f' @@@ [get_info] {selected_id} 7') 
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
                print(f' @@@ [get_info] {selected_id} 8') 
                
                return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
            else:
                
                print(f' @@@ [get_info] {selected_id} 9')
                print(f' @@@ [get_info] {selected_id} 9') 
                
                return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
    except Exception as e:
        print(f' @@@ [get_info] {selected_id} 10')
        print(f' @@@ [get_info] {selected_id} 10') 
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
            "size_human" : 0,
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
                    print(f'  GFOUND safetensors')   
                    
                    safetensors_json = vars(model_info.safetensors)
                    
                    
                    print(f'  FOUND safetensors:::::::: {safetensors_json}')
                    print(f'  GFOUND safetensors:::::::: {safetensors_json}') 
                    try:
                        quantization_key = next(iter(safetensors_json['parameters'].keys()))
                        print(f'  FOUND first key in parameters:::::::: {quantization_key}')
                        res_model_data['quantization'] = quantization_key
                        
                    except Exception as get_model_info_err:
                        print(f'  first key NOT FOUND in parameters:::::::: {quantization_key}')
                        pass
                    
                    print(f'  FOUND safetensors TOTAL :::::::: {safetensors_json["total"]}')
                    print(f'  GFOUND safetensors:::::::: {safetensors_json["total"]}')
                                        
                    print(f'  ooOOOOOOOOoooooo res_model_data["quantization"] {res_model_data["quantization"]}')
                    print(f'ooOOOOOOOOoooooo res_model_data["quantization"] {res_model_data["quantization"]}')
                    if res_model_data["quantization"] == "F32":
                        print(f'  ooOOOOOOOOoooooo found F32 -> x4')
                        print(f'ooOOOOOOOOoooooo found F32 -> x4')
                    else:
                        print(f'  ooOOOOOOOOoooooo NUUUH FIND F32 -> x2')
                        print(f'ooOOOOOOOOoooooo NUUUH FIND F32 -> x2')
                        res_model_data['size'] = int(safetensors_json["total"]) * 2
                else:
                    print(f' !!!!DIDNT FIND safetensors !!!! :::::::: ')
                    print(f' !!!!!! DIDNT FIND safetensors !!:::::::: ') 
            
            
            
            except Exception as get_model_info_err:
                res_model_data['hf_data'] = f'{get_model_info_err}'
                pass
                    
            try:
                response = requests.get(f'https://huggingface.co/{selected_id}/resolve/main/config.json', timeout=REQUEST_TIMEOUT)
                if response.status_code == 200:
                    response_json = response.json()
                    res_model_data["config_data"] = response_json
                    
                    if "architectures" in res_model_data["config_data"]:
                        res_model_data["architectures"] = res_model_data["config_data"]["architectures"][0]
                        
                    if "torch_dtype" in res_model_data["config_data"]:
                        res_model_data["torch_dtype"] = res_model_data["config_data"]["torch_dtype"]
                        print(f'  ooOOOOOOOOoooooo torch_dtype: {res_model_data["torch_dtype"]}')
                        print(f'ooOOOOOOOOoooooo torch_dtype: {res_model_data["torch_dtype"]}')
                    if "hidden_size" in res_model_data["config_data"]:
                        res_model_data["hidden_size"] = res_model_data["config_data"]["hidden_size"]
                        print(f'  ooOOOOOOOOoooooo hidden_size: {res_model_data["hidden_size"]}')
                        print(f'ooOOOOOOOOoooooo hidden_size: {res_model_data["hidden_size"]}')
                else:
                    res_model_data["config_data"] = f'{response.status_code}'
                    
            except Exception as get_config_json_err:
                res_model_data["config_data"] = f'{get_config_json_err}'
                pass                       
            
            

            res_model_data["size"], res_model_data["size_human"] = get_git_model_size(selected_id)
            
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["architectures"], res_model_data["model_id"], gr.update(value=res_model_data["size"], label=f'size ({res_model_data["size_human"]})'), res_model_data["gated"], res_model_data["model_type"], res_model_data["quantization"], res_model_data["torch_dtype"], res_model_data["hidden_size"]
        
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["model_id"], gr.update(value=res_model_data["size"], label=f'size ({res_model_data["size_human"]})'), res_model_data["gated"], res_model_data["model_type"],  res_model_data["quantization"], res_model_data["torch_dtype"], res_model_data["hidden_size"]


# def gr_load_check(selected_model_id,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_gated):
#     if selected_model_pipeline_tag != '' and selected_model_transformers == 'True' and selected_model_private == 'False' and selected_model_gated == 'False':
#         return gr.update(visible=False), gr.update(value=f'Download {selected_model_id[:12]}...', visible=True)
#     else:
#         return gr.update(visible=True), gr.update(visible=False)

def gr_load_check(selected_model_id, selected_model_architectures, selected_model_pipeline_tag, selected_model_transformers, selected_model_size, selected_model_private, selected_model_gated, selected_model_model_type, selected_model_quantization):
    

    
    # check CUDA support mit backend call
    
    # if "gguf" in selected_model_id.lower():
    #     return f'Selected a GGUF model!', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    req_model_storage = "/models"
    req_model_path = f'{req_model_storage}/{selected_model_id}'
    
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] **************** [gr_load_check] searching {selected_model_id} in {req_model_storage} (req_model_path: {req_model_path}) ...')
    print(f' **************** [gr_load_check] searching {selected_model_id} in {req_model_storage} (req_model_path: {req_model_path})...')
    


    models_found = []
    # try:                   
    #     if os.path.isdir(req_model_storage):
    #         print(f' **************** found model storage path! {req_model_storage}')
    #         print(f' **************** getting folder elements ...')       
    #         print(f' **************** found model storage path! {req_model_storage}')
    #         print(f' **************** getting folder elements ...')                        
    #         for m_entry in os.listdir(req_model_storage):
    #             m_path = os.path.join(req_model_storage, m_entry)
    #             if os.path.isdir(m_path):
    #                 for item_sub in os.listdir(m_path):
    #                     sub_item_path = os.path.join(m_path, item_sub)
    #                     models_found.append(sub_item_path)        
    #         print(f' **************** found models ({len(models_found)}): {models_found}')
    #         print(f' **************** found models ({len(models_found)}): {models_found}')
    #     else:
    #         print(f' **************** found models ({len(models_found)}): {models_found}')

    # except Exception as e:
    #     print(f' **************** ERR getting models in {req_model_storage}: {e}')


    model_path = selected_model_id
    if req_model_path in models_found:
        print(f' **************** FOUND MODELS ALREADY!!! {selected_model_id} ist in {models_found}')
        model_path = req_model_path
        return f'Model already downloaded!', gr.update(visible=True), gr.update(visible=True)
    else:
        print(f' **************** NUH UH DIDNT FIND MODEL YET!! {selected_model_id} ist NAWT in {models_found}')
    
    
        
    if selected_model_architectures == '':
        return f'Selected model has no architecture', gr.update(visible=False), gr.update(visible=False)


    # if selected_model_architectures.lower() not in defaults_frontend['vllm_supported_architectures']:
    #     if selected_model_transformers != 'True':   
    #         return f'Selected model architecture is not supported by vLLM but transformers are available (you may try to load the model in gradio Interface)', gr.update(visible=True), gr.update(visible=True)
    #     else:
    #         return f'Selected model architecture is not supported by vLLM and has no transformers', gr.update(visible=False), gr.update(visible=False)     
    
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


    return f'Selected model is supported by vLLM!'

rx_change_arr = []
def check_rx_change(current_rx_bytes):
    try:                
        try:
            int(current_rx_bytes)
        except ValueError:
            return '0'
        global rx_change_arr
        rx_change_arr += [int(current_rx_bytes)]
        if len(rx_change_arr) > 4:
            last_value = rx_change_arr[-1]
            same_value_count = 0
            for i in range(1,len(rx_change_arr)):
                if rx_change_arr[i*-1] == last_value:
                    same_value_count += 1
                    if same_value_count > 10:
                        return f'Count > 10 Download finished'
                else:
                    return f'Count: {same_value_count} {str(rx_change_arr)}'
            return f'Count: {same_value_count} {str(rx_change_arr)}'        
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'Error rx_change_arr {str(e)}'








def get_audio_path(audio_file):
    req_file = audio_file
    return [f'req_file: {req_file}', f'{req_file}']

def transcribe_audio(audio_model,audio_path,device,compute_type):  
    try:
        print(f'[transcribe_audio] audio_path ... {audio_path}')
        logging.info(f'[transcribe_audio] audio_path ... {audio_path}')
      
        AUDIO_URL = f'http://container_audio:{os.getenv("AUDIO_PORT")}/t'

        print(f'[transcribe_audio] AUDIO_URL ... {AUDIO_URL}')
        logging.info(f'[transcribe_audio] AUDIO_URL ... {AUDIO_URL}')

        print(f'[transcribe_audio] getting status ... ')
        logging.info(f'[transcribe_audio] getting status ... ')
        
        response = requests.post(AUDIO_URL, json={
            "method": "status"
        }, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:          
            print(f'[transcribe_audio] >> got response == 200 ... building json ... {response}')
            logging.info(f'[transcribe_audio] >> got response == 200 ... building json ... {response}')
            res_json = response.json()    
            print(f'[transcribe_audio] >> got res_json ... {res_json}')
            logging.info(f'[transcribe_audio] >> got res_json ... {res_json}')

            if res_json["result_data"] == "ok":
                print(f'[transcribe_audio] >> status: "ok" ... starting transcribe .... ')
                logging.info(f'[transcribe_audio] >> status: "ok" ... starting transcribe .... ')
      
                response = requests.post(AUDIO_URL, json={
                    "method": "transcribe",
                    "audio_model": audio_model,
                    "audio_path": audio_path,
                    "device": device,
                    "compute_type": compute_type
                })

                print(f'[transcribe_audio] >> got response #22222 == 200 ... building json ... {response}')
                logging.info(f'[transcribe_audio] >> got response #22222 == 200 ... building json ... {response}')
                
                res_json = response.json()
   
                print(f'[transcribe_audio] >> #22222 got res_json ... {res_json}')
                logging.info(f'[transcribe_audio] >> #22222 got res_json ... {res_json}')
                
                if res_json["result_status"] == 200:
                    return f'{res_json["result_data"]}'
                else: 
                    return 'Error :/'
            else:
                print('[transcribe_audio] ERROR AUDIO SERVER DOWN!?')
                logging.info('[transcribe_audio] ERROR AUDIO SERVER DOWN!?')
                return 'Error :/'

    except Exception as e:
        return f'Error: {e}'

def wait_for_backend(backend_url, timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.post(backend_url, json={"method": "list"}, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                print("Backend container is online.")
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass  # Backend is not yet reachable
        time.sleep(5)  # Wait for 5 seconds before retrying
    print(f"Timeout: Backend container did not come online within {timeout} seconds.")
    return False



def get_docker_container_list():
    global docker_container_list
    response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/docker', json={"method":"list"})
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
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/docker', json={"method":"logs","model":req_model})
        res_json = response.json()
        return ''.join(res_json["result_data"])
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action stop'

    
def docker_api_start(req_model):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/docker', json={"method":"start","model":req_model})
        res_json = response.json()
        return res_json
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action start {e}'

def docker_api_stop(req_model):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/docker', json={"method":"stop","model":req_model})
        res_json = response.json()
        return res_json
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action stop {e}'

def docker_api_delete(req_model):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/docker', json={"method":"delete","model":req_model})
        res_json = response.json()
        return res_json
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action delete {e}'

        
                
def toggle_compute_type(device):
    
    if device == 'cpu':
        return (
            gr.Radio(["int8"], value="int8", label="Compute type", info="Select a compute type"),
        )

    return (
        gr.Radio(["int8_float16", "float16"], value="float16", label="Compute type", info="Select a compute type"),
    )


def create_app():
    with gr.Blocks() as app:
        gr.Markdown(
        """
        # Welcome!
        Select a Hugging Face model and deploy it on a port
        
        **Note**: _[vLLM supported models list](https://docs.vllm.ai/en/latest/models/supported_models.html)_        
        """)

        input_search = gr.Textbox(placeholder="Enter Hugging Face model name or tag", label=f'found 0 models', show_label=False, autofocus=True)
        btn_search = gr.Button("Search")
        btn_tested_models = gr.Button("Load tested models")
        

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
                        selected_model_torch_dtype = gr.Textbox(label="torch_dtype")
                        selected_model_size = gr.Textbox(label="size")
                        selected_model_hidden_size = gr.Textbox(label="hidden_size", visible=False)

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
                        
        
        
        output = gr.Textbox(label="Output", lines=4, show_label=True, visible=True)
        
        
        
        
        
        
        
        
        container_state = gr.State([])   
        docker_container_list = get_docker_container_list()     
        @gr.render(inputs=container_state)
        def render_container(render_container_list):
            docker_container_list = get_docker_container_list()
            docker_container_list_sys_running = [c for c in docker_container_list if c["State"]["Status"] == "running" and c["Name"] in [f'/container_redis',f'/container_backend', f'/container_frontend', f'/container_audio']]
            docker_container_list_sys_not_running = [c for c in docker_container_list if c["State"]["Status"] != "running" and c["Name"] in [f'/container_redis',f'/container_backend', f'/container_frontend', f'/container_audio']]
            docker_container_list_vllm_running = [c for c in docker_container_list if c["State"]["Status"] == "running" and c["Name"] not in [f'/container_redis',f'/container_backend', f'/container_frontend', f'/container_audio']]
            docker_container_list_vllm_not_running = [c for c in docker_container_list if c["State"]["Status"] != "running" and c["Name"] not in [f'/container_redis',f'/container_backend', f'/container_frontend', f'/container_audio']]

            def refresh_container():
                try:
                    global docker_container_list
                    response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/docker', json={"method": "list"})
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
                    
            

            with gr.Accordion(f'System | Running {len(docker_container_list_sys_running)} | Not Running {len(docker_container_list_sys_not_running)}', open=True):
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


        
        
        


                
        with gr.Accordion(("Automatic Speech Recognition"), open=False, visible=True) as acc_audio:
            with gr.Row():
                with gr.Column(scale=2):
                    audio_input = gr.Audio(label="Upload Audio", type="filepath")
                    audio_model=gr.Dropdown(defaults_frontend['audio_models'], label="Model size", info="Select a Faster-Whisper model")
                    audio_path = gr.Textbox(visible=False)
                    device=gr.Radio(["cpu", "cuda"], value="cpu", label="Select architecture", info="Your system supports CUDA!. Make sure all drivers installed. /checkcuda if cuda")
                    compute_type=gr.Radio(["int8", "int8_float16", "float16"], value="int8", label="Compute type", info="Select a compute type")
                with gr.Column(scale=1):
                    text_output = gr.Textbox(label="Transcription", lines=8)
                    
                    transcribe_btn = gr.Button("Transcribe")
                    transcribe_btn.click(
                      get_audio_path,
                      audio_input,
                      [text_output,audio_path]
                    ).then(
                      transcribe_audio,
                      [audio_model,audio_path,device,compute_type],
                      [text_output,audio_path]
                    )
        
        
        
        device.change(
            toggle_compute_type,
            device,
            compute_type
        )
        
        
        
        
        
        input_search.change(
            search_change,
            input_search,
            [model_dropdown,input_search]
        )
        
        input_search.submit(
            search_models, 
            input_search, 
            [model_dropdown,input_search]
        ).then(
            lambda: gr.update(visible=True),
            None, 
            model_dropdown
        )
        
        btn_search.click(
            search_models, input_search, 
            [model_dropdown,input_search]
        ).then(
            lambda: gr.update(visible=True),
            None,
            model_dropdown
        )

        btn_tested_models.click(
            dropdown_load_tested_models,
            None,
            [model_dropdown,input_search]
        )

        
        model_dropdown.change(
            lambda: gr.update(visible=True), 
            None, 
            row_model_select
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            row_model_info
        ).then(
            get_info, 
            model_dropdown, 
            [selected_model_search_data,selected_model_id,selected_model_architectures,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_downloads,selected_model_container_name]
        ).then(
            get_additional_info, 
            model_dropdown, 
            [selected_model_hf_data, selected_model_config_data, selected_model_architectures,selected_model_id, selected_model_size, selected_model_gated, selected_model_model_type, selected_model_quantization, selected_model_torch_dtype, selected_model_hidden_size]
        ).then(
            gr_load_check, 
            [selected_model_id, selected_model_architectures, selected_model_pipeline_tag, selected_model_transformers, selected_model_size, selected_model_private, selected_model_gated, selected_model_model_type, selected_model_quantization],
            [output]
        )

        
        
        
        
        
        
        


    return app


# Launch the app
if __name__ == "__main__":
    backend_url = f'http://container_backend:{os.getenv("BACKEND_PORT")}/docker'
    
    
    
    # Wait for the backend container to be online
    if wait_for_backend(backend_url):
        app = create_app()
        app.launch(server_name=f'{os.getenv("FRONTEND_IP")}', server_port=int(os.getenv("FRONTEND_PORT")))
    else:
        print("Failed to start application due to backend container not being online.")
