import math
import os
import subprocess
from datetime import timedelta
from typing import List, Optional, Tuple, Union
from functools import partial
import torchaudio.compliance.kaldi as ta_kaldi
import torch.nn.functional as F
import json

import numpy as np
import requests
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip
import pickle
from tqdm import tqdm
import tempfile

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import clip
from transformers import WhisperFeatureExtractor

import sys; sys.path = ["third_party/LongVALE/"] + sys.path
try:
    from longvalellm.constants import IMAGE_TOKEN_INDEX
    from longvalellm.conversation import conv_templates, SeparatorStyle
    from longvalellm.model.builder import load_pretrained_model, load_lora
    from longvalellm.utils import disable_torch_init
    from longvalellm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
    from longvalellm.mm_utils import BEATSAudioExtractor
    from longvalellm.model.beats.BEATs import BEATs, BEATsConfig
    from longvalellm.mm_utils import VideoExtractor 
    from preprocess.beats_feature_extract import prepare_model as prepare_audio_model
    from preprocess.clip_feature_extract import prepare_model as prepare_visual_model
    from preprocess.whisper_feature_extract import prepare_model as prepare_speech_model


except ImportError:
    eval_logger.debug("LongVALE is not installed. Please install LongVALE to use this model.")

class ArgsDict:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@register_model("longvale")
class LongVALE(lmms):
    def __init__(
        self,
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        device_map="cuda:0",
        use_cache=True,
        truncate_context=False,
        max_frames_num: int = 100,
        modality="video",
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

        self.modal = modality
        self.num_frames = max_frames_num
        eval_logger.info(f"max frame nums: {self.num_frames}")

        cache_path = "data/ckpt/longvale"
        os.makedirs(cache_path, exist_ok=True)
        if not os.path.exists("data/ckpt/longvale/ViT-L-14.pt") and accelerator.is_main_process:
            subprocess.run(["wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt -O data/ckpt/longvale/ViT-L-14.pt"], shell=True)

        accelerator.wait_for_everyone()
        args = ArgsDict(
            clip_path=f"{cache_path}/ViT-L-14.pt",
            model_base=snapshot_download("lmsys/vicuna-7b-v1.5"),
            pretrain_mm_mlp_adapter=f"{cache_path}/vtimellm_stage1_mm_projector.bin",
            stage2=f"{cache_path}/longvale-vicuna-v1-5-7b-stage2-bp", 
            stage3=f"{cache_path}/longvale-vicuna-v1-5-7b-stage3-it",
            pretrain_audio_mlp_adapter=None,
            pretrain_asr_mlp_adapter=None
        )
        # processor
        self.visual_processor = prepare_visual_model("data/ckpt/longvale/ViT-L-14.pt", 0)[0]
        self.visual_processor.eval()
        self.audio_extractor = BEATSAudioExtractor(is_eval=True)
        self.audio_processor = prepare_audio_model(f"{cache_path}/BEATs_iter3_plus_AS20K.pt", 0)[0]
        self.audio_processor.eval()
        self.speech_processor = prepare_speech_model(snapshot_download("openai/whisper-large-v2"), 0)[0]
        self.speech_processor.eval()
        self.speech_transform = WhisperFeatureExtractor.from_pretrained(snapshot_download("openai/whisper-large-v2"))
        
        self._tokenizer, self._model, self.context_len = load_pretrained_model(args, args.stage2, args.stage3)
        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
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
        self._model.half()
            

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


    def get_video_chunk_content(self, video_path, max_audio_seconds=180, device="cuda:0"):
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
        
        # audio
        waveform = torch.tensor(audio_np).float().unsqueeze(0)
        waveform = waveform * 2**15
        fbank = ta_kaldi.fbank(
                waveform,
                num_mel_bins=128,
                sample_frequency=sr,
                frame_length=25,
                frame_shift=10, # 10
            )
        fbank = (fbank - 15.41663) / (2 * 6.55582)
        frame_length = 512
        fbank_pad_len = fbank.shape[0] % frame_length
        if fbank_pad_len > 0:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, fbank_pad_len))(fbank)
        curr_frames = fbank.shape[0] // frame_length
        frames = [fbank[i*frame_length:(i+1)*frame_length].unsqueeze(0) for i in range(curr_frames)]
        audio_features = torch.cat(frames, dim=0)
        audio_features = self.audio_processor.extract_features(audio_features.to(device))[0]
        audio_features = audio_features.mean(dim=1).squeeze(1)
        
        # speech 
        waveform = torch.tensor(audio_np).float()
        if len(waveform) > 30 * sr:
            audio_list = [waveform[i: i + 30 * sr] for i in range(0, len(waveform), 30 * sr)]
            spectrogram_list = []
            for audio_piece in audio_list:
                spectrogram_piece = self.speech_transform(
                    audio_piece,
                    sampling_rate=sr,
                    return_tensors="pt",
                    max_length=30 * sr,
                )
                spectrogram_list.append(spectrogram_piece["input_features"].squeeze())
            spectrogram = torch.stack(spectrogram_list, dim=0)
        else:
            spectrogram = self.speech_transform(
                waveform,
                sampling_rate=sr,
                return_tensors="pt",
                max_length=30 * sr,
                )
            spectrogram = spectrogram["input_features"].squeeze()

        features = []
        for spec in spectrogram:
            features.append(self.speech_processor(spec.unsqueeze(0).to(device), return_dict=True).last_hidden_state[0])
        features = torch.stack(features)
        dim = features.shape[-1]
        features = features.reshape(1, -1, dim)

        B, T, C = features.shape
        kernel = round(1500 * 5.12 / 30.0)
        stride = round(1500 * 5.12 / 30.0)
        kernel = (1, kernel)
        stride = (1, stride)
        speech_embeds_tr = features.transpose(1, 2).unsqueeze(2)
        speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, stride=stride)
        _, _, L = speech_embeds_overlap.shape
        speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L) 
        speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1]) 
        speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C) 
        
        speech_features = torch.mean(speech_embeds, dim=1) 

        # visual
        video_loader = VideoExtractor(N=self.num_frames)
        _, images = video_loader.extract({'id': None, 'video': video_path})
        vis_transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        images = vis_transform(images / 255.0)
        images = images.to(torch.float16)
        with torch.no_grad():
            vis_features = self.visual_processor.encode_image(images.to(device))
        
        return vis_features, audio_features, speech_features


    def generate_until(self, requests) -> List[str]:
        res = []
        output_json = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            feat_name = visuals[0].split("/")[-1].replace("mp4", "pkl")
            if os.path.exists(f"data/spatial_avqa/av_feat_pkl/longvale_feat/{feat_name}"):
                feat_data = pickle.load(open(f"data/spatial_avqa/av_feat_pkl/longvale_feat/{feat_name}", "rb"))
                vis_features, audio_features, speech_features = feat_data["vis"], feat_data["audio"], feat_data["speech"]
                vis_features = torch.from_numpy(vis_features).to("cuda")
                audio_features = torch.from_numpy(audio_features).to("cuda")
                speech_features = torch.from_numpy(speech_features).to("cuda")
            else:
                vis_features, audio_features, speech_features = self.get_video_chunk_content(visuals[0])
            
            conv = conv_templates["v1"].copy()
            conv.append_message(conv.roles[0], contexts)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            do_sample = gen_kwargs.get('do_sample', False)
            temperature = gen_kwargs.get('temperature', 0.2 if do_sample else 0.0)
            top_p = gen_kwargs.get('top_p', 0.9)
            max_new_tokens = gen_kwargs.get('max_new_tokens', 2048)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images= vis_features[None,],
                    audio = audio_features[None,], 
                    asr = speech_features[None,],
                    do_sample=do_sample,
                    temperature=temperature,
                    num_beams=1,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=max_new_tokens,
                    use_cache=True)
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            pbar.update(1)
            res.append(outputs)

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        return super().loglikelihood(requests)

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LongVALE")

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
