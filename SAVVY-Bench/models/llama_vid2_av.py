"""
VideoLLama2 evaluation code under the framework of lmms-eval - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2025-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""

import os
import subprocess
from datetime import timedelta
from typing import List, Optional, Tuple, Union
from functools import partial

import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger
from tqdm import tqdm
import pickle

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


import sys; sys.path = ["third_party/VideoLLaMA2/"] + sys.path
try:
    from videollama2.constants import (
        DEFAULT_VIDEO_TOKEN, 
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_AUDIO_TOKEN
    )
    from videollama2.model import load_pretrained_model
    from videollama2.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_image, 
        process_video, 
        process_audio_file,
        tokenizer_multimodal_token, 
    )
except ImportError:
    eval_logger.debug("VideoLLaMA2 is not installed. Please install VideoLLaMA2 to use this model.")


@register_model("llama_vid2_av")
class VideoLLaMA2(lmms):
    def __init__(
        self,
        pretrained: str = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        truncate_context=False,
        max_frames_num: int = 100,
        modality="video",
        **kwargs,
    ) -> None:
        super().__init__()

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        self.accelerator = accelerator
        self.pretrained = pretrained
        self.modal = modality
        self.model_path = snapshot_download(self.pretrained)
        self.model_name = get_model_name_from_path(pretrained)
        self.num_frames = max_frames_num
        eval_logger.info(f"max frame nums: {self.num_frames}")
        if not os.path.exists("./model_zoo/LAVIS/eva_vit_g.pth") and accelerator.is_main_process:
            eval_logger.info("\n\n Eva Encoder is not found for LLaMA-VID. Download automatically to the folder ./model_zoo/LAVIS")
            cache_path = "model_zoo/LAVIS"
            os.makedirs(cache_path, exist_ok=True)
            subprocess.run(["wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth -O ./model_zoo/LAVIS/eva_vit_g.pth"], shell=True)

        accelerator.wait_for_everyone()
        self._tokenizer, self._model, self.processor, self._max_length = load_pretrained_model(
            self.model_path,
            None,
            self.model_name,
            device_map=self.device_map,
        )
        if self._tokenizer.pad_token is None and self._tokenizer.unk_token is not None:
            self._tokenizer.pad_token = self._tokenizer.unk_token
        self.processor = {
            'image': partial(process_image, processor=self.processor, aspect_ratio=None),
            'video': partial(process_video, processor=self.processor, aspect_ratio=None, num_frames=self.num_frames),
            'audio': process_audio_file
        }

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
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
            self.accelerator = accelerator
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

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        if self.modal == 'image':
            modal_token = DEFAULT_IMAGE_TOKEN
        elif self.modal == 'video':
            modal_token = DEFAULT_VIDEO_TOKEN
        elif self.modal == 'text':
            modal_token = ''
        elif self.modal == 'audio':
            modal_token = DEFAULT_AUDIO_TOKEN
        else:
            raise ValueError(f"Unsupported modal: {self.modal}")

        
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            # 1. Vision preprocess (load & transform image or video).
            feat_dir = "data/spatial_avqa/av_feat_pkl/llamavid2_feat"
            os.makedirs(feat_dir, exist_ok=True)
            feat_name = visuals[0].split("/")[-1].replace("mp4", "pkl")
            feat_path = f"{feat_dir}/{feat_name}"
            if not os.path.exists(feat_path):
                video = self.processor[self.modal](visuals[0], va=True)
                save_feat = {}
                for key in video.keys():
                    save_feat[key] = video[key].to(torch.float32).numpy()
                pickle.dump(save_feat, open(feat_path, "wb"))
            else:
                video = pickle.load(open(feat_path, "rb"))
                for key in video.keys():
                    video[key] = torch.from_numpy(video[key]).half()
            if self.modal == 'text':
                tensor = None
            else:
                if isinstance(video, dict):
                    tensor = {k: v.half().cuda() for k, v in video.items()}
                else:
                    tensor = video.half().cuda()
            tensor = [(tensor, self.modal)]

            # 2. text preprocess (tag process & generate prompt).
            if isinstance(contexts, str):
                message = [{'role': 'user', 'content': modal_token + '\n' + contexts}]
            elif isinstance(contexts, list):
                message = copy.deepcopy(contexts)
                message[0]['content'] = modal_token + '\n' + message[0]['content']
            else:
                raise ValueError(f"Unsupported type of contexts: {type(contexts)}")
            
            if self.model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
                system_message = [
                    {'role': 'system', 'content': (
                    """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
                    """\n"""
                    """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
                    }
                ]
            else:
                system_message = []
            message = system_message + message
            prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)


            input_ids = tokenizer_multimodal_token(prompt, self.tokenizer, modal_token, return_tensors="pt").unsqueeze(0).cuda()
            attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()

            # 3. generate response according to visual signals and prompts. 
            keywords = [self.tokenizer.eos_token]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            do_sample = gen_kwargs.get('do_sample', False)
            temperature = gen_kwargs.get('temperature', 0.2 if do_sample else 0.0)
            top_p = gen_kwargs.get('top_p', 0.9)
            max_new_tokens = gen_kwargs.get('max_new_tokens', 2048)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    images=tensor,
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            pbar.update(1)
            res.append(outputs)

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        return super().loglikelihood(requests)

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for VideoLLaMA2_AV")
    
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
