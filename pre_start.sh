#!/bin/bash

export PYTHONUNBUFFERED=1
source /venv/bin/activate
#rsync -au --remove-source-files /ComfyUI/ /workspace/ComfyUI/
#ln -s /comfy-models/* /workspace/ComfyUI/models/checkpoints/

cd /workspace/ComfyUI
git pull
CUDA_VISIBLE_DEVICES=1
python main.py --listen --port 3000 &
