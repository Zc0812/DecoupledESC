#launch Vanilla Model
MODEL_PATH_VM =  #the path to the model
PORT = 8000
API_BASE = "http://localhost:$PORT/v1"
API_KEY = "EMPTY"
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH_VM  \
    --dtype bfloat16 --max_model_len 8192 --port $PORT \