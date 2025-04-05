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

res_container = {
  "INFO": [
    "Application startup complete.",
    "Uvicorn running on http://0.0.0.0:7861 (Press CTRL+C to quit)"
  ],
  "containers": [
    {
      "container": "all",
      "info": {
        "name": "/dummy_container",
        "id": "0000000000000000000000000000000000000000000000000000000000000000",
        "read": "2025-01-01T00:00:00.000000000Z",
        "preread": "2025-01-01T00:00:00.000000000Z",
        "pids_stats": {
          "current": 0,
          "limit": 0
        },
        "blkio_stats": {
          "io_service_bytes_recursive": "0",
          "io_serviced_recursive": "0",
          "io_queue_recursive": "0",
          "io_service_time_recursive": "0",
          "io_wait_time_recursive": "0",
          "io_merged_recursive": "0",
          "io_time_recursive": "0",
          "sectors_recursive": "0"
        },
        "num_procs": 0,
        "storage_stats": {},
        "cpu_stats": {
          "cpu_usage": {
            "total_usage": 0,
            "usage_in_kernelmode": 0,
            "usage_in_usermode": 0
          },
          "system_cpu_usage": 0,
          "online_cpus": 0,
          "throttling_data": {
            "periods": 0,
            "throttled_periods": 0,
            "throttled_time": 0
          }
        },
        "precpu_stats": {
          "cpu_usage": {
            "total_usage": 0,
            "usage_in_kernelmode": 0,
            "usage_in_usermode": 0
          },
          "system_cpu_usage": 0,
          "online_cpus": 0,
          "throttling_data": {
            "periods": 0,
            "throttled_periods": 0,
            "throttled_time": 0
          }
        },
        "memory_stats": {
          "usage": 0,
          "stats": {
            "active_anon": 0,
            "active_file": 0,
            "anon": 0,
            "anon_thp": 0,
            "file": 0,
            "file_dirty": 0,
            "file_mapped": 0,
            "file_writeback": 0,
            "inactive_anon": 0,
            "inactive_file": 0,
            "kernel_stack": 0,
            "pgactivate": 0,
            "pgdeactivate": 0,
            "pgfault": 0,
            "pglazyfree": 0,
            "pglazyfreed": 0,
            "pgmajfault": 0,
            "pgrefill": 0,
            "pgscan": 0,
            "pgsteal": 0,
            "shmem": 0,
            "slab": 0,
            "slab_reclaimable": 0,
            "slab_unreclaimable": 0,
            "sock": 0,
            "thp_collapse_alloc": 0,
            "thp_fault_alloc": 0,
            "unevictable": 0,
            "workingset_activate": 0,
            "workingset_nodereclaim": 0,
            "workingset_refault": 0
          },
          "limit": 0
        },
        "networks": {
          "eth0": {
            "rx_bytes": 0,
            "rx_packets": 0,
            "rx_errors": 0,
            "rx_dropped": 0,
            "tx_bytes": 0,
            "tx_packets": 0,
            "tx_errors": 0,
            "tx_dropped": 0
          }
        }
      },
      "current_dl": "0.00 MBit/s (total: 558)",
      "timestamp": "2025-03-08 00:39:03"
    },
    {
      "container": "container_frontend",
      "info": {
        "name": "/container_frontend",
        "id": "dd8cd00638baf6443594ad4ba5975381ae4da4938ba044f8deb4ae85017804ed",
        "read": "2025-03-08T00:39:04.781135085Z",
        "preread": "2025-03-08T00:39:03.778649876Z",
        "pids_stats": {
          "current": 15,
          "limit": 60398
        },
        "blkio_stats": {
          "io_service_bytes_recursive": "0",
          "io_serviced_recursive": "0",
          "io_queue_recursive": "0",
          "io_service_time_recursive": "0",
          "io_wait_time_recursive": "0",
          "io_merged_recursive": "0",
          "io_time_recursive": "0",
          "sectors_recursive": "0"
        },
        "num_procs": 0,
        "storage_stats": {},
        "cpu_stats": {
          "cpu_usage": {
            "total_usage": 2407450000,
            "usage_in_kernelmode": 171465000,
            "usage_in_usermode": 2235985000
          },
          "system_cpu_usage": 242375940000000,
          "online_cpus": 15,
          "throttling_data": {
            "periods": 0,
            "throttled_periods": 0,
            "throttled_time": 0
          }
        },
        "precpu_stats": {
          "cpu_usage": {
            "total_usage": 493075000,
            "usage_in_kernelmode": 55784000,
            "usage_in_usermode": 437291000
          },
          "system_cpu_usage": 242360920000000,
          "online_cpus": 15,
          "throttling_data": {
            "periods": 0,
            "throttled_periods": 0,
            "throttled_time": 0
          }
        },
        "memory_stats": {
          "usage": 238092288,
          "stats": {
            "active_anon": 230776832,
            "active_file": 0,
            "anon": 230821888,
            "anon_thp": 0,
            "file": 1523712,
            "file_dirty": 1523712,
            "file_mapped": 0,
            "file_writeback": 0,
            "inactive_anon": 0,
            "inactive_file": 1523712,
            "kernel_stack": 245760,
            "pgactivate": 0,
            "pgdeactivate": 0,
            "pgfault": 63925,
            "pglazyfree": 0,
            "pglazyfreed": 0,
            "pgmajfault": 0,
            "pgrefill": 0,
            "pgscan": 0,
            "pgsteal": 0,
            "shmem": 0,
            "slab": 3599136,
            "slab_reclaimable": 3137528,
            "slab_unreclaimable": 461608,
            "sock": 0,
            "thp_collapse_alloc": 0,
            "thp_fault_alloc": 0,
            "unevictable": 0,
            "workingset_activate": 0,
            "workingset_nodereclaim": 0,
            "workingset_refault": 0
          },
          "limit": 52784435200
        },
        "networks": {
          "eth0": {
            "rx_bytes": 752,
            "rx_packets": 10,
            "rx_errors": 0,
            "rx_dropped": 0,
            "tx_bytes": 84,
            "tx_packets": 2,
            "tx_errors": 0,
            "tx_dropped": 0
          }
        }
      },
      "current_dl": "000000000000000",
      "timestamp": "2025-03-08 00:39:04"
    },
    {
      "container": "container_backend",
      "info": {
        "name": "/container_backend",
        "id": "ee915ffe402a7e3b566899fd04a5f9c632ca642c26016bdc7db417613dfb0ee0",
        "read": "2025-03-08T00:39:06.785085952Z",
        "preread": "2025-03-08T00:39:05.782821461Z",
        "pids_stats": {
          "current": 2,
          "limit": 60398
        },
        "blkio_stats": {
          "io_service_bytes_recursive": "0",
          "io_serviced_recursive": "0",
          "io_queue_recursive": "0",
          "io_service_time_recursive": "0",
          "io_wait_time_recursive": "0",
          "io_merged_recursive": "0",
          "io_time_recursive": "0",
          "sectors_recursive": "0"
        },
        "num_procs": 0,
        "storage_stats": {},
        "cpu_stats": {
          "cpu_usage": {
            "total_usage": 673737000,
            "usage_in_kernelmode": 90502000,
            "usage_in_usermode": 583235000
          },
          "system_cpu_usage": 242405990000000,
          "online_cpus": 15,
          "throttling_data": {
            "periods": 0,
            "throttled_periods": 0,
            "throttled_time": 0
          }
        },
        "precpu_stats": {
          "cpu_usage": {
            "total_usage": 673737000,
            "usage_in_kernelmode": 90502000,
            "usage_in_usermode": 583235000
          },
          "system_cpu_usage": 242390960000000,
          "online_cpus": 15,
          "throttling_data": {
            "periods": 0,
            "throttled_periods": 0,
            "throttled_time": 0
          }
        },
        "memory_stats": {
          "usage": 65626112,
          "stats": {
            "active_anon": 62664704,
            "active_file": 0,
            "anon": 62672896,
            "anon_thp": 0,
            "file": 860160,
            "file_dirty": 860160,
            "file_mapped": 0,
            "file_writeback": 0,
            "inactive_anon": 0,
            "inactive_file": 860160,
            "kernel_stack": 32768,
            "pgactivate": 0,
            "pgdeactivate": 0,
            "pgfault": 21873,
            "pglazyfree": 0,
            "pglazyfreed": 0,
            "pgmajfault": 0,
            "pgrefill": 0,
            "pgscan": 0,
            "pgsteal": 0,
            "shmem": 0,
            "slab": 1801544,
            "slab_reclaimable": 1539040,
            "slab_unreclaimable": 262504,
            "sock": 0,
            "thp_collapse_alloc": 0,
            "thp_fault_alloc": 0,
            "unevictable": 0,
            "workingset_activate": 0,
            "workingset_nodereclaim": 0,
            "workingset_refault": 0
          },
          "limit": 52784435200
        },
        "networks": {
          "eth0": {
            "rx_bytes": 1322,
            "rx_packets": 19,
            "rx_errors": 0,
            "rx_dropped": 0,
            "tx_bytes": 126,
            "tx_packets": 3,
            "tx_errors": 0,
            "tx_dropped": 0
          }
        }
      },
      "current_dl": "000000000000000",
      "timestamp": "2025-03-08 00:39:06"
    },
    {
      "container": "container_vllm",
      "info": {
        "name": "/container_vllm",
        "id": "1ec72d9ddca02d724d60b912af78b8bf54318ebd27037c58e26a7cd674f41e69",
        "read": "2025-03-08T00:39:08.789481597Z",
        "preread": "2025-03-08T00:39:07.786951635Z",
        "pids_stats": {
          "current": 19,
          "limit": 60398
        },
        "blkio_stats": {
          "io_service_bytes_recursive": [
            {
              "major": 253,
              "minor": 0,
              "op": "read",
              "value": 0
            },
            {
              "major": 253,
              "minor": 0,
              "op": "write",
              "value": 2170880
            }
          ],
          "io_serviced_recursive": "0",
          "io_queue_recursive": "0",
          "io_service_time_recursive": "0",
          "io_wait_time_recursive": "0",
          "io_merged_recursive": "0",
          "io_time_recursive": "0",
          "sectors_recursive": "0"
        },
        "num_procs": 0,
        "storage_stats": {},
        "cpu_stats": {
          "cpu_usage": {
            "total_usage": 4958235000,
            "usage_in_kernelmode": 1286219000,
            "usage_in_usermode": 3672016000
          },
          "system_cpu_usage": 242435970000000,
          "online_cpus": 15,
          "throttling_data": {
            "periods": 0,
            "throttled_periods": 0,
            "throttled_time": 0
          }
        },
        "precpu_stats": {
          "cpu_usage": {
            "total_usage": 4957435000,
            "usage_in_kernelmode": 1286011000,
            "usage_in_usermode": 3671423000
          },
          "system_cpu_usage": 242421020000000,
          "online_cpus": 15,
          "throttling_data": {
            "periods": 0,
            "throttled_periods": 0,
            "throttled_time": 0
          }
        },
        "memory_stats": {
          "usage": 404107264,
          "stats": {
            "active_anon": 387440640,
            "active_file": 0,
            "anon": 387440640,
            "anon_thp": 0,
            "file": 2170880,
            "file_dirty": 0,
            "file_mapped": 0,
            "file_writeback": 0,
            "inactive_anon": 0,
            "inactive_file": 2170880,
            "kernel_stack": 311296,
            "pgactivate": 0,
            "pgdeactivate": 0,
            "pgfault": 122268,
            "pglazyfree": 0,
            "pglazyfreed": 0,
            "pgmajfault": 0,
            "pgrefill": 0,
            "pgscan": 0,
            "pgsteal": 0,
            "shmem": 0,
            "slab": 11875384,
            "slab_reclaimable": 11206888,
            "slab_unreclaimable": 668496,
            "sock": 0,
            "thp_collapse_alloc": 0,
            "thp_fault_alloc": 0,
            "unevictable": 0,
            "workingset_activate": 0,
            "workingset_nodereclaim": 0,
            "workingset_refault": 0
          },
          "limit": 52784435200
        },
        "networks": {
          "eth0": {
            "rx_bytes": 1560,
            "rx_packets": 24,
            "rx_errors": 0,
            "rx_dropped": 0,
            "tx_bytes": 126,
            "tx_packets": 3,
            "tx_errors": 0,
            "tx_dropped": 0
          }
        }
      },
      "current_dl": "000000000000000",
      "timestamp": "2025-03-08 00:39:08"
    },
    {
      "container": "container_redis",
      "info": {
        "name": "/container_redis",
        "id": "78fd33fe7b6904c2e9fe3b614098cba262916be9271762088f28b9bda652d5c1",
        "read": "2025-03-08T00:39:10.793574274Z",
        "preread": "2025-03-08T00:39:09.791041164Z",
        "pids_stats": {
          "current": 6,
          "limit": 60398
        },
        "blkio_stats": {
          "io_service_bytes_recursive": "0",
          "io_serviced_recursive": "0",
          "io_queue_recursive": "0",
          "io_service_time_recursive": "0",
          "io_wait_time_recursive": "0",
          "io_merged_recursive": "0",
          "io_time_recursive": "0",
          "sectors_recursive": "0"
        },
        "num_procs": 0,
        "storage_stats": {},
        "cpu_stats": {
          "cpu_usage": {
            "total_usage": 50150000,
            "usage_in_kernelmode": 18945000,
            "usage_in_usermode": 31204000
          },
          "system_cpu_usage": 242466030000000,
          "online_cpus": 15,
          "throttling_data": {
            "periods": 0,
            "throttled_periods": 0,
            "throttled_time": 0
          }
        },
        "precpu_stats": {
          "cpu_usage": {
            "total_usage": 48547000,
            "usage_in_kernelmode": 18064000,
            "usage_in_usermode": 30483000
          },
          "system_cpu_usage": 242451000000000,
          "online_cpus": 15,
          "throttling_data": {
            "periods": 0,
            "throttled_periods": 0,
            "throttled_time": 0
          }
        },
        "memory_stats": {
          "usage": 3633152,
          "stats": {
            "active_anon": 3006464,
            "active_file": 0,
            "anon": 3006464,
            "anon_thp": 0,
            "file": 0,
            "file_dirty": 0,
            "file_mapped": 0,
            "file_writeback": 0,
            "inactive_anon": 0,
            "inactive_file": 0,
            "kernel_stack": 98304,
            "pgactivate": 0,
            "pgdeactivate": 0,
            "pgfault": 2719,
            "pglazyfree": 0,
            "pglazyfreed": 0,
            "pgmajfault": 0,
            "pgrefill": 0,
            "pgscan": 0,
            "pgsteal": 0,
            "shmem": 0,
            "slab": 350048,
            "slab_reclaimable": 135560,
            "slab_unreclaimable": 214488,
            "sock": 0,
            "thp_collapse_alloc": 0,
            "thp_fault_alloc": 0,
            "unevictable": 0,
            "workingset_activate": 0,
            "workingset_nodereclaim": 0,
            "workingset_refault": 0
          },
          "limit": 52784435200
        },
        "networks": {
          "eth0": {
            "rx_bytes": 2624,
            "rx_packets": 38,
            "rx_errors": 0,
            "rx_dropped": 0,
            "tx_bytes": 658,
            "tx_packets": 11,
            "tx_errors": 0,
            "tx_dropped": 0
          }
        }
      },
      "current_dl": "000000000000000",
      "timestamp": "2025-03-08 00:39:10"
    }
  ],
  "gpu_data": [
    {
      "gpu_i": 0,
      "gpu_info": "{'gpu_i': 0, 'current_uuid': 'GPU-5119e922-1797-1ace-ec58-cc0a2772f5ee', 'gpu_util': 0.0, 'mem_used': 403.75, 'mem_total': 8192.0, 'mem_util': 4.9285888671875}",
      "timestamp": "2025-03-08 00:39:10"
    }
  ],
  "network_data": [
    {
      "container": "all",
      "info": "all",
      "current_dl": "0.00 MBit/s (total: 558)",
      "timestamp": "2025-03-08 00:39:03"
    },
    {
      "container": "container_frontend",
      "info": "container_frontend",
      "current_dl": "000000000000000",
      "timestamp": "2025-03-08 00:39:04"
    },
    {
      "container": "container_backend",
      "info": "container_backend",
      "current_dl": "000000000000000",
      "timestamp": "2025-03-08 00:39:06"
    },
    {
      "container": "container_vllm",
      "info": "container_vllm",
      "current_dl": "000000000000000",
      "timestamp": "2025-03-08 00:39:08"
    }
  ]
}

current_models_data = []
db_gpu_data = []
db_gpu_data_len = ''

GLOBAL_SEARCH_INPUT_TS = 0
GLOBAL_SEARCH_INPUT_THRESHOLD = 10
GLOBAL_SEARCH_REQUEST_TIMEOUT = 3
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








def welcome(name):
    if name == "car":
        return f"meep meep"
    return f"Welcome to Gradio, {name}!"

def update_components(input_text):
    # Count the number of characters in the input
    char_count = len(input_text)
    
    # Create a list of components to return
    components = []
    
    # Add the output textbox first
    output = gr.Textbox(value=welcome(input_text))
    components.append(output)
    
    # Add additional buttons based on character count
    buttons = []
    for i in range(char_count):
        buttons.append(
            gr.Button(value=f"Button {i+1} for '{input_text[i]}'")
        )
    
    # Return both the output and all buttons
    return [output, *buttons]



model_dropdown = gr.Dropdown(choices=[''], label=f'Select a Hugging Face model', interactive=True, show_label=False, visible=False)

        
        
def dropdown_load_tested_models():
    tested_models = [
        {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        },
        {
        "id": "PowerInfer/SmallThinker-3B-Preview",
        },
        {
        "id": "bigcode/starcoder2-3b",
        },
        {
        "id": "bigcode/starcoder2-7b",
        },
        {
        "id": "ibm-granite/granite-3.0-1b-a400m-base",
        },
        {
        "id": "stabilityai/stablelm-3b-4e1t",
        },
        {
        "id": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        },
        {
        "id": "adept/persimmon-8b-chat",
        },
        {
        "id": "adept/persimmon-8b-base",
        },
        {
        "id": "allenai/OLMoE-1B-7B-0924-Instruct",
        },
        {
        "id": "facebook/opt-125m",
        }
    ]
    global current_models_data
    response_models = tested_models
    print(f'response_models: {response_models}')
    current_models_data = response_models.copy()
    model_ids = [m["id"] for m in response_models]
    print(f'model_ids: {model_ids}')
    # return gr.update(choices=model_ids, value=response_models[0]["id"], visible=True)
    return [gr.update(choices=model_ids, value=response_models[0]["id"], visible=True),gr.update(value=response_models[0]["id"],show_label=True, label=f'Loaded {len(model_ids)} models!')]

def huggingface_hub_search(query):
    try:
        global current_models_data
        response = requests.get(f'https://huggingface.co/api/models?search={query}', timeout=GLOBAL_SEARCH_REQUEST_TIMEOUT)
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
        response = requests.get(f'https://huggingface.co/api/models?search={query}', timeout=GLOBAL_SEARCH_REQUEST_TIMEOUT)
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
                response = requests.get(f'https://huggingface.co/{selected_id}/resolve/main/config.json', timeout=GLOBAL_SEARCH_REQUEST_TIMEOUT)
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








def transcribe_audio(audio_file):
    req_file = audio_file
    return f'req_file: {req_file}'



REQUEST_TIMEOUT = 300
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
        
        
        
        
        with gr.Accordion(("Automatic Speech Recognition"), open=False, visible=True) as acc_audio:
            with gr.Row():
                with gr.Column(scale=2):
                    audio_input = gr.Audio(label="Upload Audio", type="filepath")
                
                with gr.Column(scale=1):
                    text_output = gr.Textbox(label="Transcription", lines=8)
                
                    transcribe_btn = gr.Button("Transcribe")
                    transcribe_btn.click(
                        transcribe_audio,
                        inputs=audio_input,
                        outputs=text_output
                    )
        
        
        
        
        
        
        
        
        
        input_search.change(
            search_change,
            input_search,
            [model_dropdown,input_search]
        )
        
        input_search.submit(search_models, inputs=input_search, outputs=[model_dropdown,input_search]).then(lambda: gr.update(visible=True), None, model_dropdown)
        btn_search.click(search_models, inputs=input_search, outputs=[model_dropdown,input_search]).then(lambda: gr.update(visible=True), None, model_dropdown)

        btn_tested_models.click(
            dropdown_load_tested_models,
            None,
            [model_dropdown,input_search]
        )
        with gr.Row():
            port_model = gr.Number(value=8001,visible=False,label="Port of model: ")
            port_vllm = gr.Number(value=8000,visible=False,label="Port of vLLM: ")
        
        info_textbox = gr.Textbox(value="Interface not possible for selected model. Try another model or check 'pipeline_tag', 'transformers', 'private', 'gated'", show_label=False, visible=False)
        btn_dl = gr.Button("Download", visible=False)
        
        # model_dropdown.change(get_info, model_dropdown, [selected_model_search_data,selected_model_id,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_downloads],selected_model_container_name).then(get_additional_info, model_dropdown, [selected_model_hf_data, selected_model_config_data, selected_model_id, selected_model_size, selected_model_gated]).then(lambda: gr.update(visible=True), None, selected_model_pipeline_tag).then(lambda: gr.update(visible=True), None, selected_model_transformers).then(lambda: gr.update(visible=True), None, selected_model_private).then(lambda: gr.update(visible=True), None, selected_model_downloads).then(lambda: gr.update(visible=True), None, selected_model_size).then(lambda: gr.update(visible=True), None, selected_model_gated).then(lambda: gr.update(visible=True), None, port_model).then(lambda current_value: current_value + 1, port_model, port_model).then(lambda: gr.update(visible=True), None, port_vllm).then(lambda current_value: current_value + 1, port_vllm, port_vllm).then(gr_load_check, [selected_model_id,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_gated],[info_textbox,btn_dl])
        
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

        create_response = gr.Textbox(label="Building container...", show_label=True, visible=False)  
        timer_dl_box = gr.Textbox(label="Dowmload progress:", visible=False)
        
        btn_interface = gr.Button("Load Interface",visible=False)
        @gr.render(inputs=[selected_model_pipeline_tag, selected_model_id], triggers=[btn_interface.click])
        def show_split(text_pipeline, text_model):
            if len(text_model) == 0:
                gr.Markdown("Error pipeline_tag or model_id")
            else:
                gr.Interface.from_pipeline(pipeline(text_pipeline, model=text_model))


        btn_dl.click(lambda: gr.update(label="Building vLLM container",visible=True), None, create_response).then(lambda: gr.update(visible=True), None, timer_dl_box).then(lambda: gr.update(visible=True), None, btn_interface)
        gr.Markdown(
        """
        # Dynamic Buttons Demo!
        Start typing below and watch buttons appear based on your input length.
        """)
        
        
        
        
        
        
        
        
        
        
        
        with gr.Row():
            inp = gr.Textbox(placeholder="Type something here", label="Main Input")
            out = gr.Textbox(label="Output")
        
        # This will hold our dynamic buttons
        button_group = gr.Group()
        
        # Update both the output and dynamic buttons when input changes
        inp.change(
            fn=update_components,
            inputs=inp,
            outputs=[out, button_group]
        )

    return app


# Launch the app
if __name__ == "__main__":
    backend_url = f'http://{os.getenv("CONTAINER_BACKEND")}:{os.getenv("BACKEND_PORT")}/docker'
    
    # Wait for the backend container to be online
    if wait_for_backend(backend_url):
        app = create_app()
        app.launch(server_name=f'{os.getenv("FRONTEND_IP")}', server_port=int(os.getenv("FRONTEND_PORT")))
    else:
        print("Failed to start application due to backend container not being online.")
