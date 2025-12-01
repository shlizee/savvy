#!/bin/bash
cd third_party/
git clone https://github.com/EvolvingLMMs-Lab/EgoLife.git
cd ../

cp models/egogpt.py third_party/lmms_eval/lmms_eval/models/egogpt.py
cp -r tasks/spatial_avqa third_party/lmms_eval/lmms_eval/tasks/
pip install -r requirements/egogpt_requirements.txt

cd third_party/lmms_eval
pip install -e .
cd ../../

cp third_party/EgoGPT_SAVVY/speech_encoder.py third_party/EgoLife/EgoGPT/egogpt/model/speech_encoder/speech_encoder.py

bash scripts/eval_model_base.sh  --model egogpt_7b --num_processes 4 --benchmark spatial_avqa --max_frame_num 32