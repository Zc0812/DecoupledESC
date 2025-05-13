#launch Strategy Planner
MODEL_PATH_SP =  #the path to the model
PORT = 8000
API_BASE = "http://localhost:$PORT/v1"
API_KEY = "EMPTY"
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH_SP  \
    --dtype bfloat16 --max_model_len 8192 --port $PORT \