#!/bin/bash
cp models/minicpm_o.py third_party/lmms_eval/lmms_eval/models/minicpm_o.py
cp -r tasks/spatial_avqa third_party/lmms_eval/lmms_eval/tasks/
pip install -r requirements/minicpm_requirements.txt
pip install --no-cache-dir vocos

cd third_party/lmms_eval
pip install -e .
cd ../../

bash scripts/eval_model_base.sh  --model minicpm_o_8b --num_processes 4 --benchmark spatial_avqa --max_frame_num 32