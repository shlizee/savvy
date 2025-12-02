"""
Adaptation of the framework of lmms-eval to SAVVY-Bench benchmarking code - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""

# Developed based upon lmms-eval from https://github.com/EvolvingLMMs-Lab/lmms-eval. Below is the original copyright:
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import importlib
import os
import sys

import hf_transfer
from loguru import logger

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stdout, level="WARNING")

AVAILABLE_MODELS = {
    "aero": "Aero",
    "plm": "PerceptionLM",
    "aria": "Aria",
    "auroracap": "AuroraCap",
    "batch_gpt4": "BatchGPT4",
    "claude": "Claude",
    "cogvlm2": "CogVLM2",
    "from_log": "FromLog",
    "fuyu": "Fuyu",
    "gemini_api": "GeminiAPI",
    "gpt4v": "GPT4V",
    "idefics2": "Idefics2",
    "instructblip": "InstructBLIP",
    "internvideo2": "InternVideo2",
    "internvl": "InternVLChat",
    "internvl2": "InternVL2",
    "llama_vid": "LLaMAVid",
    "llama_vision": "LlamaVision",
    "llava": "Llava",
    "llava_hf": "LlavaHf",
    "llava_onevision": "Llava_OneVision",
    "llava_onevision_moviechat": "Llava_OneVision_MovieChat",
    "llava_sglang": "LlavaSglang",
    "llava_vid": "LlavaVid",
    "longva": "LongVA",
    "mantis": "Mantis",
    "minicpm_v": "MiniCPM_V",
    "minimonkey": "MiniMonkey",
    "moviechat": "MovieChat",
    "mplug_owl_video": "mplug_Owl",
    "ola": "Ola",
    "openai_compatible": "OpenAICompatible",
    "oryx": "Oryx",
    "phi3v": "Phi3v",
    "phi4_multimodal": "Phi4",
    "qwen2_5_omni": "Qwen2_5_Omni",
    "qwen2_5_vl": "Qwen2_5_VL",
    "qwen2_5_vl_interleave": "Qwen2_5_VL_Interleave",
    "qwen2_audio": "Qwen2_Audio",
    "qwen2_vl": "Qwen2_VL",
    "qwen_vl": "Qwen_VL",
    "qwen_vl_api": "Qwen_VL_API",
    "reka": "Reka",
    "ross": "Ross",
    "slime": "Slime",
    "srt_api": "SRT_API",
    "tinyllava": "TinyLlava",
    "videoChatGPT": "VideoChatGPT",
    "videochat2": "VideoChat2",
    "videollama3": "VideoLLaMA3",
    "video_llava": "VideoLLaVA",
    "vila": "VILA",
    "vita": "VITA",
    "vllm": "VLLM",
    "xcomposer2_4KHD": "XComposer2_4KHD",
    "xcomposer2d5": "XComposer2D5",
    "egogpt": "EgoGPT",
    "internvideo2_5": "InternVideo2_5",
    "videochat_flash": "VideoChat_Flash",
    "whisper": "Whisper",
    "whisper_vllm": "WhisperVllm",
    "vora": "VoRA",
    "llama_vid2_av": "VideoLLaMA2",
    ### SAVVY-Bench Eval modifications: add 6 AV-LLMs here"
    "minicpm_o": "MiniCPM_O",
    "salmonn_vid": "SALMONNVid",
    "egogpt": "EgoGPT",
    "ola": "OLA",
    "longvale": "LongVALE"
    ####
}


def get_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")

    model_class = AVAILABLE_MODELS[model_name]
    if "." not in model_class:
        model_class = f"lmms_eval.models.{model_name}.{model_class}"

    try:
        model_module, model_class = model_class.rsplit(".", 1)
        module = __import__(model_module, fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise


if os.environ.get("LMMS_EVAL_PLUGINS", None):
    # Allow specifying other packages to import models from
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        m = importlib.import_module(f"{plugin}.models")
        for model_name, model_class in getattr(m, "AVAILABLE_MODELS").items():
            AVAILABLE_MODELS[model_name] = f"{plugin}.models.{model_name}.{model_class}"
