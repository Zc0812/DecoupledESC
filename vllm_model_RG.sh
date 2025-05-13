#launch Response Generator
MODEL_PATH_RG =  #the path to the model
PORT = 8001
API_BASE = "http://localhost:$PORT/v1"
API_KEY = "EMPTY" 
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_PATH_RG  \
    --dtype bfloat16 --max_model_len 8192 --port $PORT \