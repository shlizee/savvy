#!/bin/bash
cp models/gemini_api.py third_party/lmms_eval/lmms_eval/models/gemini_api.py
cp -r tasks/spatial_avqa third_party/lmms_eval/lmms_eval/tasks/

cd third_party/lmms_eval
pip install -e .
cd ../../
pip install google-generativeai
bash scripts/eval_model_base.sh  --model gemini_flash --num_processes 4 --benchmark spatial_avqa --max_frame_num 32