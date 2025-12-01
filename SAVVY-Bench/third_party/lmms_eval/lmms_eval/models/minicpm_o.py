import warnings
import math
import pickle
import os
import numpy as np
from typing import List, Optional, Tuple, Union
import json

import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from PIL import Image
import tempfile
from tqdm import tqdm
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip
import logging

from transformers import AutoModel, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

@register_model("minicpm_o")
class MiniCPM_O(lmms):
    """
    MiniCPM_O Model
    """

    def __init__(
        self,
        pretrained: str = "openbmb/MiniCPM-o-2_6",
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        max_frames_num: int = 100,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device
        self._model = AutoModel.from_pretrained(pretrained, trust_remote_code=trust_remote_code, torch_dtype=dtype, device_map=self._device, attn_implementation='sdpa').to(dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self._config = self._model.config
        self.num_frames = max_frames_num
        self.model.eval()
        self.model.tie_weights()
        self.batch_size_per_gpu = int(batch_size)
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
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "We have not implemented this function for MiniCPM_V yet"

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_video_chunk_content(self, video_path, max_audio_seconds=180):
        video = VideoFileClip(video_path)
        fps = video.fps
        sr = 16000
        if video.audio:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                temp_audio_file_path = temp_audio_file.name
                video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=sr, verbose=False, logger=None)
                audio_np, sr = librosa.load(temp_audio_file_path, sr=sr, mono=True)
        else:
            audio_np = np.zeros((round(sr * video.duration)))
        
        
        contents = []
        valid_count = 0
        for i in range(self.num_frames):
            # Get frame at uniform positions
            frame_time = min(video.duration - 1/fps, (i + 1) * video.duration / self.num_frames)
            try:
                frame = video.get_frame(frame_time)
                image = Image.fromarray((frame).astype(np.uint8))
                contents.extend([image])
                valid_count += 1
            except:
                continue
        # Add the processed audio to the contents
        contents.extend([audio_np])
        
        return contents


    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        output_json = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            contents = []
            visual = visuals[0] # support one video
            feat_dir = "data/spatial_avqa/av_feat_pkl/minicpm_feat"
            os.makedirs(feat_dir, exist_ok=True)
            feat_name = visual.split("/")[-1].replace("mp4", "pkl")
            feat_path = f"{feat_dir}/{feat_name}"
            if not os.path.exists(feat_path):   
                contents = self.get_video_chunk_content(visual)
                pickle.dump(contents, open(feat_path, "wb"))
            else:
                contents = pickle.load(open(feat_path, "rb"))
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            assert len(visuals) == 1, "MiniCPM_O interface does not support bn_image > 1 for now"
            
            sys_msg = self.model.get_sys_prompt(mode='omni', language='en')
            if "<image>" in contexts:
                contexts = contexts.replace("<image>", "")
            
            msgs = [{"role": "user", "content": contents+[contexts]}]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.2
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            try:
                # ominicpm does not give much information on how they do eval so I just use the chat format.
                outputs = self.model.chat(
                    image=None,
                    msgs=msgs,
                    context=None,
                    tokenizer=self.tokenizer,
                    use_image_id=False,
                    max_slice_nums=1,
                    do_sample=gen_kwargs["do_sample"],
                    temperature=gen_kwargs["temperature"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
            pbar.update(1)
            res.append(outputs.strip())
            
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        return super().loglikelihood(requests)

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for MiniCPM")
    
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
