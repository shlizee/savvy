import math
import os
import subprocess
from datetime import timedelta
from typing import List, Optional, Tuple, Union
from functools import partial
import argparse
import yaml
import json
from omegaconf import OmegaConf
import json

import pickle
from PIL import Image
import tempfile
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip

import numpy as np
import requests
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav
from lmms_eval.utils import stop_sequences_criteria
try:
    from lmms_eval.models.salmonn_configs.openllama import OpenLLAMAPEFTModel
    from lmms_eval.models.salmonn_configs.sft_dataset_nomix import *
except ImportError:
    eval_logger.debug("video_SALMONN is not installed. Please install video_SALMONN to use this model.")


@register_model("salmonn_vid")
class SALMONNVid(lmms):
    def __init__(
        self,
        arg_file_path = "",
        pretrained: str = "tsinghua-ee/SALMONN-13B",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = False,
        revision=None,
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        truncate_context=False,
        num_frames: int = 100,
        modality="video",
        max_frames_num: int = 100,
        **kwargs,
    ) -> None:
        super().__init__()

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.max_num_frames = max_frames_num
        print(self.max_num_frames)
        
        self.args = OmegaConf.load(arg_file_path)
        self._model = OpenLLAMAPEFTModel(**self.args)
        
        delta_ckpt = torch.load(self.args['delta_ckpt_path'], map_location=torch.device('cpu'))
        self._model.load_state_dict(delta_ckpt, strict=False)

        accelerator.wait_for_everyone()
        self.processor = {
            'audio': WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2", cache_dir=self.args["cache_dir"])
        }

        self.model.eval()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    def download_file(self, url, folder_path):
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Extract filename from URL
        filename = url.split("/")[-1]

        # Define path to save the file
        file_path = os.path.join(folder_path, filename)

        # Send a GET request to the URL
        response = requests.get(url)

        # Check if request was successful (status code 200)
        if response.status_code == 200:
            # Save the file to the specified folder
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"File downloaded successfully to {file_path}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list


    def get_audio(self, audio, sample_rate):
        if len(audio.shape) == 2:
            audio = audio[:, 0]
        if audio.shape[0] < 3 * sample_rate:
            audio = np.concatenate((audio, np.zeros((3 * sample_rate - audio.shape[0]), dtype=float)), axis=0)
        if len(audio) > 30 * sample_rate:
            audio_list = [audio[i: i + 30 * sample_rate] for i in range(0, len(audio), 30 * sample_rate)]
            spectrogram_list = []
            for audio_piece in audio_list:
                spectrogram_piece = self.processor['audio'](
                    audio_piece,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    max_length=30 * sample_rate,
                )
                spectrogram_list.append(spectrogram_piece["input_features"].squeeze())
            spectrogram = torch.stack(spectrogram_list, dim=0)
            return spectrogram, audio_list

    def get_video_chunk_content(self, video_path, flatten=True):
        sr = 16000
        video = VideoFileClip(video_path)
        if video.audio:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                temp_audio_file_path = temp_audio_file.name
                video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=sr, verbose=False, logger=None)
                audio_np, sr = librosa.load(temp_audio_file_path, sr=sr, mono=True)
        else:
            audio_np = np.zeros((round(sr * video.duration)))
        
        spectrogram, raw_audios = self.get_audio(audio_np, sr)

        video = EncodedVideo.from_path(
            video_path,
            decoder="pyav",
            decode_audio=False,
        )
        
        frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=self.max_num_frames)
        start_sec = 0
        end_sec = video.duration
        
        # Extract the frames - this returns a tensor with shape [C, T, H, W]
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_clip = video_data["video"]  # This should have a shape attribute
        
        video_clip = frame_sampler(video_clip)
        video_clip = video_clip / 255.0
        
        return {"audio": spectrogram, "video": video_clip, "raw_audios": raw_audios}

    def generate_until(self, requests) -> List[str]:
        res = []
        output_json = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            videos = []
            audios = []
            raw_audios = []
            for visual in visuals:
                # 1. Vision preprocess (load & transform image or video).
                video_dict = self.get_video_chunk_content(visual)
                videos.append(video_dict["video"].cuda())
                audios.append(video_dict["audio"].cuda())
                video_dict["raw_audios"][-1] = np.pad(video_dict["raw_audios"][-1], (0, 480000 - len(video_dict["raw_audios"][-1])), 'constant', constant_values=0)
                raw_audios.append(video_dict["raw_audios"])

            with torch.inference_mode():
                outputs = self.model(dict(
                    image_paths=[audios, videos],
                    output_texts=[contexts],
                    modality='audiovideoimage',
                    audiomasks=torch.tensor([1, 1]).cuda(),
                    raw_audios=raw_audios
                ), generate=True, generate_config=gen_kwargs)
            
            outputs = outputs[0][0].strip()

            pbar.update(1)
            res.append(outputs)

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        return super().loglikelihood(requests)

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for VideoSalmonn")

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size
