"""
EGOGPT evaluation code under the framework of lmms-eval - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""

import copy
import json
import logging
import math
import re
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union
import json
import pickle
import numpy as np
import PIL
import torch
import transformers
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from tqdm import tqdm
from transformers import AutoConfig
import torch.distributed as dist
from fairscale.nn.model_parallel import initialize as fs_init

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True

import sys; sys.path = ["third_party/EgoLife/EgoGPT/"] + sys.path
# Import LLaVA modules
try:
    import copy
    import os
    import re
    import sys
    import warnings

    import numpy as np
    import requests
    import soundfile as sf
    import torch
    import whisper
    from decord import VideoReader, cpu
    from egogpt.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_SPEECH_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
        SPEECH_TOKEN_INDEX,
    )
    from egogpt.conversation import SeparatorStyle, conv_templates
    from egogpt.mm_utils import get_model_name_from_path, process_images
    from egogpt.model.builder import load_pretrained_model
    from PIL import Image
    from scipy.signal import resample
except ImportError as e:
    eval_logger.debug(f"egogpt is not installed. Please install egogpt to use this model.\nError: {e}")


# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("egogpt")
class EgoGPT(lmms):
    """
    EgoGPT Model
    """

    def __init__(
        self,
        pretrained: str = "checkpoints/egogpt_IT_12k_1126_zero3",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "qwen_1_5",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = "decord",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        dist.init_process_group(
            backend="nccl", rank=0, world_size=1,
            init_method=f"tcp://127.0.0.1:23560")
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

        egogpt_model_args = {}
        if attn_implementation is not None:
            egogpt_model_args["attn_implementation"] = attn_implementation

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend
        # Try to load the model with the multimodal argument
        self._tokenizer, self._model, self._max_length = load_pretrained_model(pretrained, device_map=self.device_map, **egogpt_model_args)
        self._image_processor = self._model.get_vision_tower().image_processor
        self._config = self._model.config
        self.model.eval()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1

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
            self._world_size = 1

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

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

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
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for EgoGPT")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def split_text(self, text, keywords):
        pattern = "(" + "|".join(map(re.escape, keywords)) + ")"
        parts = re.split(pattern, text)
        parts = [part for part in parts if part]
        return parts

    def load_video(self, video_path=None, audio_path=None, max_frames_num=16, fps=1, task_name=None):
        audio_path = video_path.replace("mp4", "pkl")
        sr = 16000
        if os.path.exists(audio_path):
            # print(f"Loading existing audio from {audio_path}")
            speech = pickle.load(open(audio_path, "rb"))
        else:
            from moviepy.editor import VideoFileClip
            import tempfile
            import librosa
            video = VideoFileClip(video_path)
            if video.audio:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                    temp_audio_file_path = temp_audio_file.name
                    video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=sr, verbose=False, logger=None)
                    speech, sr = librosa.load(temp_audio_file_path, sr=sr, mono=True)
            else:
                import pdb; pdb.set_trace()
                speech = np.zeros((round(sr * video.duration)))
            pickle.dump(speech, open(audio_path, 'wb'))
        
        sample_rate = 16000

        # Calculate max samples for 30 seconds
        max_audio_length = 30 * sample_rate  # 30 seconds at 16kHz = 480,000 samples
        
        # Create list of audio chunks, each no more than 30 seconds
        speech_chunks = []
        for i in range(0, len(speech), max_audio_length):
            chunk = speech[i:i + max_audio_length]
            
            # Process each chunk
            chunk = whisper.pad_or_trim(chunk.astype(np.float32))
            chunk = whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0)
            
            speech_chunks.append(chunk)
        
        # Create a list of lengths for each chunk
        speech_lengths = torch.LongTensor([chunk.shape[0] for chunk in speech_chunks])

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        avg_fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
        frame_time = [i / avg_fps for i in frame_idx]

        if max_frames_num > 0:
            if len(frame_idx) > max_frames_num:
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
        if task_name == "egoplan":
            # add current ovservation frame
            frame_idx.append(total_frame_num - 1)
        video = vr.get_batch(frame_idx).asnumpy()
        return video, speech_chunks, speech_lengths

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        output_json = []
        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []
            for visual, context in zip(batched_visuals, batched_contexts):
                if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                    self._config.image_aspect_ratio = origin_image_aspect_ratio
                    eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

                feat_dir = "data/spatial_avqa/av_feat_pkl/egogpt_feat"
                os.makedirs(feat_dir, exist_ok=True)
                feat_name = visual[0].split("/")[-1].replace("mp4", "pkl")
                feat_path = f"{feat_dir}/{feat_name}"
                if not os.path.exists(feat_path):
                    if visual is None or visual == []:  # for text-only tasks.
                        visual = None
                        task_type = "text"
                        placeholder_count = 0
                        image_tensor = None
                    else:
                        if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                            self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                            eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                        if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:  # overwrite logic for video task with multiple static image frames
                            assert type(visual) == list, "sample_frames must be specified for video task"
                            sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                            visual = [visual[i] for i in sample_indices]
                            assert len(visual) == metadata["sample_frames"]

                            image_tensor = process_images(visual, self._image_processor, self._config)
                            if type(image_tensor) is list:
                                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                            else:
                                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                            image_tensor = [image_tensor]
                            task_type = "video"
                            placeholder_count = 1

                        elif type(visual[0]) == PIL.Image.Image:  # For image, multi-image tasks
                            image_tensor = process_images(visual, self._image_processor, self._config)
                            speech = torch.zeros(3000, 128)
                            speech_lengths = torch.LongTensor([3000])
                            if type(image_tensor) is list:
                                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                            else:
                                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                            task_type = "image"
                            placeholder_count = len(visual) if isinstance(visual, list) else 1

                        elif type(visual[0]) == str:  # For video task
                            image_tensor = []
                            try:
                                if self.video_decode_backend == "decord":
                                    if "egoplan" in visual[0]:
                                        task_name = "egoplan"
                                    else:
                                        task_name = None
                                    frames, speech_chunks, speech_lengths = self.load_video(video_path=visual[0], max_frames_num=self.max_frames_num, task_name=task_name)
                                else:
                                    raise NotImplementedError("Only decord backend is supported for video task")
                                processed_frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                                processed_frames = processed_frames.half()
                                image_tensor.append(processed_frames)
                                image_sizes = [frames[0].size]
                            except Exception as e:
                                eval_logger.error(f"Error {e} in loading video")
                                image_tensor = None

                            task_type = "video"
                            placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                    save_feat = {
                        "image_tensor": image_tensor,
                        "task_type": task_type,
                        "speech_lengths": speech_lengths,
                        "speech_chunks": speech_chunks,
                        "frames": frames
                    }
                    pickle.dump(save_feat, open(feat_path, "wb"))

                else:
                    save_feat = pickle.load(open(feat_path, "rb"))
                    image_tensor = save_feat["image_tensor"]
                    task_type = save_feat["task_type"]
                    speech_lengths = save_feat["speech_lengths"]
                    speech_chunks = save_feat["speech_chunks"]
                    frames = save_feat["frames"]

                if DEFAULT_IMAGE_TOKEN not in context:
                    question = DEFAULT_IMAGE_TOKEN + "\n" + context
                else:
                    question = context
                speech = torch.stack(speech_chunks).to(self.device).half()
                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()

                if utils.is_json(question):  # conversational question input
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item["value"]
                        conv.append_message(role, message)

                    assert len(conv.messages) % 2 == 1
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:  # only simple string for question
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            # preconfigure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            parts = self.split_text(prompt_question, ["<image>", "<speech>"])
            input_ids = []
            for part in parts:
                if "<image>" == part:
                    input_ids += [IMAGE_TOKEN_INDEX]
                elif "<speech>" == part:
                    input_ids += [SPEECH_TOKEN_INDEX]
                else:
                    input_ids += self.tokenizer(part).input_ids

            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            input_ids_list = [input_ids]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            input_ids = torch.tensor(input_ids, dtype=torch.long).squeeze(0).to(self.device)
            if task_type == "image":
                gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
            elif task_type == "video":
                gen_kwargs["modalities"] = ["video"]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode
                gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

            if "image_aspect_ratio" in gen_kwargs.keys():
                gen_kwargs.pop("image_aspect_ratio")
            try:
                with torch.inference_mode():
                    image_tensor = [image_tensor_item for image_tensor_item in image_tensor]
                    cont = self.model.generate(input_ids, images=image_tensor, speech=speech, speech_lengths=speech_lengths, **gen_kwargs)

                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            except Exception as e:
                raise e

            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("generate_until_multi_round is not implemented for EgoGPT")