#!/bin/bash
cd third_party/
git clone https://github.com/DAMO-NLP-SG/VideoLLaMA2
cd VideoLLaMA2
git checkout audio_visual
cd ../../

cp models/llama_vid2_av.py third_party/lmms_eval/lmms_eval/models/llama_vid2_av.py
cp -r tasks/spatial_avqa third_party/lmms_eval/lmms_eval/tasks/
cp third_party/VideoLLaMA2_SAVVY/videollama2/mm_utils.py third_party/VideoLLaMA2/videollama2/mm_utils.py
pip install -r requirements/videollama_requirements.txt

cd third_party/lmms_eval
pip install -e .
cd ../../

bash scripts/eval_model_base.sh  --model video_llama2_7b --num_processes 4 --benchmark spatial_avqa --max_frame_num 32