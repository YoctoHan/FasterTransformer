import argparse
import dataclasses
import os
import pathlib
import typing

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

str_type_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

class StarCoderWeights:
    def __init__(self,        head_num, size_per_head, layer_num,          vocab_size,
                 max_seq_len, tensor_para_size,        pipeline_para_size,
                 weights_data_type: typing.Union[str, np.dtype],
                 inference_data_type: str,
                 ):
        assert(head_num % tensor_para_size == 0)

        self.head_num            =                        head_num
        self.size_per_head       =                   size_per_head
        self.layer_num           =                       layer_num
        self.vocab_size          =                      vocab_size
        self.max_seq_len         =                     max_seq_len
        self.tensor_para_size    =                tensor_para_size
        self.pipeline_para_size  =              pipeline_para_size
        self.layers_per_device   = layer_num // pipeline_para_size

        local_head_num           =    head_num // tensor_para_size
        global_head_num          =                        head_num
        local_hidden_units       =  local_head_num * size_per_head
        global_hidden_units      = global_head_num * size_per_head
        local_inter_size         =          local_hidden_units * 4

        self.local_head_num      =                  local_head_num
        self.global_head_num     =                 global_head_num
        self.local_hidden_units  =              local_hidden_units
        self.global_hidden_units =             global_hidden_units
        self.local_inter_size    =                local_inter_size

        self.share_embed = False

        if isinstance(weights_data_type, str):
            try:
                weights_data_type = {
                    "fp16"    : np.float16,
                    "fp32"    : np.float32,
                    "float16" : np.float16,
                    "float32" : np.float32,
                }[weights_data_type]
            except KeyError:
                raise ValueError(f"Don't know how to interpret weights_data_type: {weights_data_type}")

        assert weights_data_type in [np.float32, np.float16]
        self.weights_data_type   =   weights_data_type
        self.inference_data_type = inference_data_type

        self.weight = []
        # Transformer blocks
        self.weight.extend([torch.zeros(         6144, dtype = torch.float16)] * (self.layer_num))   # input_layernorm.weight
        self.weight.extend([torch.zeros(         6144, dtype = torch.float16)] * (self.layer_num))   # input_layernorm.bias
        self.weight.extend([torch.zeros([ 6144, 6144], dtype = torch.float16)] * (self.layer_num))   # attention.query.weight
        self.weight.extend([torch.zeros([  256, 6144], dtype = torch.float16)] * (self.layer_num))   # attention.key_value.weight
        self.weight.extend([torch.zeros(         6144, dtype = torch.float16)] * (self.layer_num))   # attention.query.bias
        self.weight.extend([torch.zeros(          256, dtype = torch.float16)] * (self.layer_num))   # attention.key_value.bias
        self.weight.extend([torch.zeros([ 6144, 6144], dtype = torch.float16)] * (self.layer_num))   # attention.dense.weight
        self.weight.extend([torch.zeros(         6144, dtype = torch.float16)] * (self.layer_num))   # attention.dense.bias
        self.weight.extend([torch.zeros(         6144, dtype = torch.float16)] * (self.layer_num))   # post_attention_layernorm.weight
        self.weight.extend([torch.zeros(         6144, dtype = torch.float16)] * (self.layer_num))   # post_attention_layernorm.bias
        self.weight.extend([torch.zeros([24576, 6144], dtype = torch.float16)] * (self.layer_num))   # mlp.dense_h_to_4h.weight
        self.weight.extend([torch.zeros(        24576, dtype = torch.float16)] * (self.layer_num))   # mlp.dense_h_to_4h.bias
        self.weight.extend([torch.zeros([6144, 24576], dtype = torch.float16)] * (self.layer_num))   # mlp.dense_4h_to_h.weight
        self.weight.extend([torch.zeros(         6144, dtype = torch.float16)] * (self.layer_num))   # mlp.dense_4h_to_h.bias
        
        # Final layernorm
        self.weight.extend([torch.zeros(         6144, dtype = torch.float16)])          # final_layernorm.weight
        self.weight.extend([torch.zeros(         6144, dtype = torch.float16)])          # final_layernorm.bias
        
        # Embedding blocks
        self.weight.extend([torch.zeros([49152, 6144], dtype = torch.float16)])          # word_embeddings
        self.weight.extend([torch.zeros([ 8192, 6144], dtype = torch.float16)])          # position_embeddings

    def __getitem__(self, idx):
        return self.weight[idx]

    def __setitem__(self, idx, val):
        self.weight[idx] = val

    def __len__(self):
        return len(self.weight)

    def _map(self, func):
        for i in range(len(self.weight)):
            if isinstance(self.weight[i], list):
                for j in range(len(self.weight[i])):
                    self.weight[i][j] = func(self.weight[i][j])
            else:
                self.weight[i] = func(self.weight[i])

    def load(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Failed to find {ckpt_path}")
        else :
            print(" -Loading weights- ".center(119, "="))
            
        weight    =                                                                    []
        module    = torch.load(ckpt_path, map_location='cpu')['module']['language_model']

        weight.extend([module['transformer'][         f'layers.{i}.input_layernorm.weight'].to(torch.float16) for i in range(self.layer_num)])
        weight.extend([module['transformer'][           f'layers.{i}.input_layernorm.bias'].to(torch.float16) for i in range(self.layer_num)])
        weight.extend([module['transformer'][         f'layers.{i}.attention.query.weight'].to(torch.float16) for i in range(self.layer_num)])
        weight.extend([module['transformer'][     f'layers.{i}.attention.key_value.weight'].to(torch.float16) for i in range(self.layer_num)])
        weight.extend([module['transformer'][           f'layers.{i}.attention.query.bias'].to(torch.float16) for i in range(self.layer_num)])
        weight.extend([module['transformer'][       f'layers.{i}.attention.key_value.bias'].to(torch.float16) for i in range(self.layer_num)])
        weight.extend([module['transformer'][         f'layers.{i}.attention.dense.weight'].to(torch.float16) for i in range(self.layer_num)])
        weight.extend([module['transformer'][           f'layers.{i}.attention.dense.bias'].to(torch.float16) for i in range(self.layer_num)])
        weight.extend([module['transformer'][f'layers.{i}.post_attention_layernorm.weight'].to(torch.float16) for i in range(self.layer_num)])
        weight.extend([module['transformer'][  f'layers.{i}.post_attention_layernorm.bias'].to(torch.float16) for i in range(self.layer_num)])
        weight.extend([module['transformer'][       f'layers.{i}.mlp.dense_h_to_4h.weight'].to(torch.float16) for i in range(self.layer_num)])
        weight.extend([module['transformer'][         f'layers.{i}.mlp.dense_h_to_4h.bias'].to(torch.float16) for i in range(self.layer_num)])
        weight.extend([module['transformer'][       f'layers.{i}.mlp.dense_4h_to_h.weight'].to(torch.float16) for i in range(self.layer_num)])
        weight.extend([module['transformer'][         f'layers.{i}.mlp.dense_4h_to_h.bias'].to(torch.float16) for i in range(self.layer_num)])

        weight.extend([module['transformer'][                    f'final_layernorm.weight'].to(torch.float16)])
        weight.extend([module['transformer'][                      f'final_layernorm.bias'].to(torch.float16)])

        weight.extend([module[  'embedding'][    f'word_embeddings'][            f'weight'].to(torch.float16)])
        weight.extend([module[  'embedding'][f'position_embeddings'][            f'weight'].to(torch.float16)])

        def weright_reshape(weight, self_weight) -> None:
            for i in range(len(weight)):
                if weight[i].nelement() > 0:
                    try:
                        self_weight[i] = weight[i].reshape(self_weight[i].shape)
                    except:
                        raise RuntimeError(f"shape error: {weight[i].shape} => {self_weight[i].shape}")

        weright_reshape(weight, self.weight)
        print(" -Loading weights DONE- ".center(119, "="))
        return True

class StarCoder(nn.Module):
    def __init__(self,
                 head_num,                    size_per_head,
                 vocab_size,                  start_id,      end_id, layer_num,
                 max_seq_len                : int,
                 tensor_para_size           : int,
                 pipeline_para_size         : int,
                 lib_path                   : typing.Union[str, pathlib.Path],
                 inference_data_type        : str,
                 inter_size                 : int = 0,
                 # gpt_variant_params
                 layernorm_eps              : float = 1e-6,
                 layernorm_type             : typing.Literal['pre_layernorm', 'post_layernorm'] = "pre_layernorm",
                 activation_type            : str = "Gelu",
                 gpt_with_moe               : bool = False,
                 expert_num                 : int = 0,
                 moe_k                      : int = 0,
                 moe_layer_index            : typing.List = [],
                 has_positional_encoding    : bool = True,
                 has_pre_decoder_layernorm  : bool = False,
                 has_post_decoder_layernorm : bool = True,
                 has_adapters               : bool = False,
                 adapter_inter_size         : int = 0,
                 use_attention_linear_bias  : bool = False,
                 int8_mode                  : int = 0,
                 weights_data_type          : typing.Union[str, np.dtype] = np.float32,
                 shared_contexts_ratio      : float = 1.0):
        super().__init__()
        self.head_num      =      head_num
        self.size_per_head = size_per_head
        self.inter_size    =    inter_size
        self.vocab_size    =    vocab_size
        self.start_id      =      start_id
        self.end_id        =        end_id
        self.layer_num     =     layer_num

        # multi-gpu params
        self.tensor_para_size   =   tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.use_sparse_gemm    =              False
        self.build_model        =              False
        self.weights_data_type  =  weights_data_type
        
        # star_coder_variant_params
        self.layernorm_eps = layernorm_eps
        self.layernorm_type = layernorm_type
        self.activation_type = activation_type
        self.gpt_with_moe = gpt_with_moe
        self.expert_num = expert_num
        self.moe_k = moe_k
        self.moe_layer_index = moe_layer_index
        self.has_positional_encoding = has_positional_encoding
        self.has_pre_decoder_layernorm = has_pre_decoder_layernorm
        self.has_post_decoder_layernorm = has_post_decoder_layernorm
        self.has_adapters = has_adapters
        self.adapter_inter_size = adapter_inter_size
        self.use_attention_linear_bias = use_attention_linear_bias

        assert torch.cuda.is_available(), "CUDA is required for this model."

        assert head_num  %   tensor_para_size == 0, "head_num must be a multiple of tensor_para_size."
        assert layer_num % pipeline_para_size == 0, "layer_num must be a multiple of pipeline_para_size."

        # Load the C++ model into Pytorch model.
        print(" -Loading the C++ model into Pytorch model- ".center(119, "="))
        torch.classes.load_library(os.path.abspath(lib_path))
        print(" -Load the C++ model into Pytorch model DONE- ".center(119, "="))

        # Prepare weights
        print(" -Preparing weights- ".center(119, "="))
        self.weights = StarCoderWeights(head_num,    size_per_head,    layer_num,          vocab_size,
                                        max_seq_len, tensor_para_size, pipeline_para_size,
                                        weights_data_type   =   weights_data_type,
                                        inference_data_type = inference_data_type)
        print(" -Prepare weights DONE- ".center(119, "="))
        # Prepare for tensor/pipeline parallel
        try:
            dist.init_process_group(backend='mpi')
        except:
            print("[INFO] WARNING: Have initialized the process group")
        self.rank         =               dist.get_rank()
        self.device_count =     torch.cuda.device_count()
        self.device       = self.rank % self.device_count
        torch.cuda.set_device(self.device)

        world_size = dist.get_world_size()
        assert world_size == tensor_para_size * pipeline_para_size, \
                             "tensor_para_size * pipeline_para_size must be equal to world_size."

        self.tensor_para_rank   = self.rank  % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size

    def load(self, ckpt_path):
        is_load = self.weights.load(ckpt_path)
        self.cuda()
        torch.cuda.empty_cache()  # clean cache for model weight preprocessing
        return is_load

    def cuda(self):
        self.weights._map(lambda weights: weights.cuda(self.device))

        if self.build_model:
            del self.model
            self.build_model = False

        self.model = torch.classes.FasterTransformer.StarCoderOp(
            self.head_num, self.size_per_head, self.inter_size,
            self.layer_num,
            self.vocab_size, self.start_id, self.end_id,
            self.use_sparse_gemm,
            # gpt_variant_params
            self.layernorm_eps,
            self.layernorm_type,
            self.activation_type,
            self.has_positional_encoding,
            self.has_pre_decoder_layernorm,
            self.has_post_decoder_layernorm,
            self.has_adapters,
            self.adapter_inter_size,
            self.use_attention_linear_bias,
            self.weights.weight)
        self.build_model = True
        
    def forward(self,
                start_ids: torch.IntTensor,
                start_lengths: torch.IntTensor,
                output_len: int,
                beam_width: int = 1,
                top_k: typing.Optional[torch.IntTensor] = None,
                top_p: typing.Optional[torch.FloatTensor] = None,
                beam_search_diversity_rate: typing.Optional[torch.FloatTensor] = None,
                temperature: typing.Optional[torch.FloatTensor] = None,
                len_penalty: typing.Optional[torch.FloatTensor] = None,
                repetition_penalty: typing.Optional[torch.FloatTensor] = None,
                presence_penalty: typing.Optional[torch.FloatTensor] = None,
                min_length: typing.Optional[torch.IntTensor] = None,
                random_seed: typing.Optional[torch.LongTensor] = None,
                bad_words_list: typing.Optional[torch.IntTensor] = None,
                return_output_length: bool = False,
                return_cum_log_probs: int = 0):
        if not self.build_model:
            # for the cases we don't load model
            self.cuda()
            torch.cuda.empty_cache()  # clean cache for model weight preprocessing
        input_len = start_ids.size(1)
        assert input_len > 0, "input len must be larger than zero. For an unconditional case, use start_id as the first token."

        # Inputs to device
        start_ids = start_ids.cuda(self.device)
        start_lengths = start_lengths.cuda(self.device)
        # outputs: output_ids, output_lengths, output_cum_log_probs (optional) 
        outputs = self.model.forward(start_ids,
                                     start_lengths,
                                     output_len,
                                     beam_width,  # optional, can be None
                                     top_k,  # optional, can be None
                                     top_p,  # optional, can be None
                                     beam_search_diversity_rate,  # optional, can be None
                                     temperature,  # optional, can be None
                                     len_penalty,  # optional, can be None
                                     repetition_penalty,  # optional, can be None
                                     random_seed,  # optional, can be None
                                     return_cum_log_probs)  # optional, can be None
        if return_cum_log_probs == 0:
            output_ids, output_lengths = outputs
        else:
            output_ids, output_lengths, output_cum_log_probs = outputs
        if return_output_length:
            if return_cum_log_probs > 0:
                return output_ids, output_lengths, output_cum_log_probs
            else:
                return output_ids, output_lengths
        else:
            return output_ids

    # def set_input_tensor(self, input_tensor):
    #     """Set input tensor to be used instead of forward()'s input.

    #     When doing pipeline parallelism the input from the previous
    #     stage comes from communication, not from the input, so the
    #     model's forward_step_func won't have it. This function is thus
    #     used by internal code to bypass the input provided by the
    #     forward_step_func"""
    #     self.input_tensor = input_tensor


# @dataclasses.dataclass
# class GptInitModelParameters:
#     head_num: int
#     size_per_head: int
#     layer_num: int
#     max_seq_len: int
#     tensor_para_size: int
#     vocab_size: int
#     start_id: int
#     end_id: int
#     pipeline_para_size: int
#     weights_data_type: str
#     has_adapters: bool
#     adapter_inter_size: int
#     data_type: str
#     int8_mode: int
#     sparse: int
#     # GPT variant params.
#     layernorm_eps: float = 1e-6
#     layernorm_type: typing.Literal['pre_layernorm', 'post_layernorm'] = 'pre_layernorm'
#     activation_type: str = 'gelu'
#     has_positional_encoding: bool = True
#     has_pre_decoder_layernorm: bool = False
#     has_post_decoder_layernorm: bool = True
#     use_attention_linear_bias: bool = False
#     inter_size: int = 0

#     PREDEFINED_MODELS: typing.ClassVar[dict] = {
#         'default': dict(),
#         'opt-pre': dict(layernorm_eps=1e-5,
#                         layernorm_type='pre_layernorm',
#                         activation_type='relu',
#                         has_post_decoder_layernorm=True),
#         'opt-pre': dict(layernorm_eps=1e-5,
#                         layernorm_type='post_layernorm',
#                         activation_type='relu',
#                         has_post_decoder_layernorm=False),
#         'bloom': dict(layernorm_eps=1e-5,
#                       layernorm_type='pre_layernorm',
#                       activation_type='gelu',
#                       has_positional_encoding=False,
#                       has_pre_decoder_layernorm=True,
#                       has_post_decoder_layernorm=True,
#                       use_attention_linear_bias=True)
#     }

#     def gpt_init_kwargs(self):
#         do_not_include = ["sparse", "data_type"]
#         args = {k: v for k, v in dataclasses.asdict(self).items() if k not in do_not_include}
#         args["inference_data_type"] = dataclasses.asdict(self)["data_type"]
#         return args

#     @classmethod
#     def from_args(cls, args, config_reader):
#         model_name = args.model_name
#         head_num = config_reader.getint(model_name, "head_num")
#         size_per_head = config_reader.getint(model_name, "size_per_head")
#         param = cls(
#             head_num=head_num,
#             size_per_head=size_per_head,
#             layer_num=config_reader.getint(model_name, "num_layer"),
#             # There is no limitation on the length when no positional encoding,
#             # setting by a large enough integer.
#             max_seq_len=config_reader.getint(model_name, "max_pos_seq_len", fallback=int(1e7)),
#             tensor_para_size=config_reader.getint(model_name, "tensor_para_size"),
#             vocab_size=config_reader.getint(model_name, "vocab_size"),
#             start_id=config_reader.getint(model_name, "start_id"),
#             end_id=config_reader.getint(model_name, "end_id"),
#             weights_data_type=config_reader.get(model_name, "weight_data_type"),
#             has_adapters=config_reader.getboolean(model_name, "has_adapters", fallback=False),
#             adapter_inter_size=config_reader.getint(model_name, "adapter_inter_size", fallback=0),
#             pipeline_para_size=(
#                 args.pipeline_para_size
#                 or config_reader.getint("ft_instance_hyperparameter", "pipeline_para_size", fallback=1)
#             ),
#             int8_mode=(
#                 args.int8_mode
#                 if args.int8_mode is not None
#                 else config_reader.getint("ft_instance_hyperparameter", "int8_mode", fallback=0)
#             ),
#             data_type=(
#                 args.data_type or
#                 config_reader.get("ft_instance_hyperparameter", "data_type",
#                                   fallback=config_reader.get(model_name, "weight_data_type"))
#             ),
#             sparse=int(getattr(args, 'sparse', False)),
#             inter_size=config_reader.getint(model_name, "inter_size", fallback=4*head_num*size_per_head)
#         )

#         if config_reader.has_option(model_name, 'model_variant'):
#             model_type = config_reader.get(model_name, 'model_variant')
#             model_params = cls.PREDEFINED_MODELS[model_type]
#             param.update(model_params)

#         return param

#     def update(self, update_params: dict):
#         for k, v in update_params:
#             setattr(self, k, v)
#         return self

#     def asdict(self):
#         return dataclasses.asdict(self)

#     @classmethod
#     def update_argparser(cls, parser):
#         parser.add_argument("--model-name", type=str, default="gpt", help="Model name from config.ini file")
#         parser.add_argument("--pipeline-para-size", type=int, help="size of pipeline parallelism")
#         parser.add_argument("--data-type", type=str, help="data type", choices=["fp32", "bf16", "fp16"])
#         parser.add_argument(
#             "--sparse", action='store_true',
#             help="Enable sparse matrix multiplication. (Need SM 8.0 or 8.6 and SPARSITY_SUPPORT=ON)")
#         parser.add_argument("--int8-mode", type=int, choices=[0, 1], help="Set int8 mode")


# @dataclasses.dataclass
# class GptRuntimeModelParameters:
#     beam_width: int
#     top_k: torch.Tensor
#     top_p: torch.Tensor
#     beam_search_diversity_rate: torch.Tensor
#     temperature: torch.Tensor
#     len_penalty: torch.Tensor
#     repetition_penalty: torch.Tensor

#     def gpt_forward_kwargs(self):
#         return dataclasses.asdict(self)

#     @classmethod
#     def from_args(cls, args, config_reader, batch_size=None):
#         bs = args.batch_size
#         if batch_size is not None:
#             bs = batch_size
#         return cls(
#             beam_width=args.beam_width or config_reader.getint("ft_instance_hyperparameter", "beam_width", fallback=1),
#             top_k=(args.sampling_top_k or config_reader.getint("ft_instance_hyperparameter", "top_k", fallback=1)) *
#             torch.ones(size=[bs], dtype=torch.int32),
#             top_p=(args.sampling_top_p or config_reader.getfloat("ft_instance_hyperparameter", "top_p", fallback=0.0)) *
#             torch.ones(size=[bs], dtype=torch.float32),
#             beam_search_diversity_rate=(
#                 args.beam_search_diversity_rate
#                 or config_reader.getfloat("ft_instance_hyperparameter", "beam_search_diversity_rate", fallback=0.0)
#             ) * torch.ones(size=[bs], dtype=torch.float32),
#             temperature=(args.temperature or config_reader.getfloat("ft_instance_hyperparameter",
#                          "temperature", fallback=1.0)) * torch.ones(size=[bs], dtype=torch.float32),
#             len_penalty=(args.len_penalty or config_reader.getfloat("ft_instance_hyperparameter",
#                          "len_penalty", fallback=0.0)) * torch.ones(size=[bs], dtype=torch.float32),
#             repetition_penalty=(
#                 args.repetition_penalty or config_reader.getfloat("ft_instance_hyperparameter", "repetition_penalty", fallback=1.0)
#             ) * torch.ones(size=[bs], dtype=torch.float32),
#         )

#     def slice_args(self, idx):
#         return GptRuntimeModelParameters(
#             beam_width=self.beam_width,
#             top_k=self.top_k[idx],
#             top_p=self.top_p[idx],
#             beam_search_diversity_rate=self.beam_search_diversity_rate[idx],
#             temperature=self.temperature[idx],
#             len_penalty=self.len_penalty[idx],
#             repetition_penalty=self.repetition_penalty[idx],
#         )

#     @classmethod
#     def update_argparser(cls, parser):
#         parser.add_argument("--beam-width", type=int, help="beam width")
#         parser.add_argument("--sampling-top-k", type=int, help="Candidate (k) value of top k sampling in decoding")
#         parser.add_argument("--sampling-top-p", type=float, help="Probability (p) value of top p sampling in decoding.")
#         parser.add_argument("--temperature", type=float, help="temperature")
#         parser.add_argument("--len-penalty", type=float, help="len_penalty")
#         parser.add_argument("--repetition-penalty", type=float, help="repetition penalty")
#         parser.add_argument("--beam-search-diversity-rate", type=float, help="beam_search_diversity_rate")


# DEFAULT_START_TAG = "<|endoftext|>"
# DEFAULT_END_TAG = "<|endoftext|>"
# OPENAI_GPT2_START_ID = 50256
# OPENAI_GPT2_END_ID = 50256


# @dataclasses.dataclass
# class GptModelConfig:
#     model_name: str
#     tensor_para_size: int
#     head_num: int
#     size_per_head: int
#     inter_size: int
#     num_layer: int
#     max_pos_seq_len: int
#     weight_data_type: str
#     vocab_size: int
#     start_id: int
#     end_id: int

#     @classmethod
#     def from_nemo_package(
#         cls,
#         *,
#         args: argparse.Namespace,
#         nemo_model_config: typing.Dict[str, typing.Any],
#         bos_id: int,
#         eos_id: int,
#         vocab_size: int,
#     ):

#         return cls(
#             model_name="gpt",
#             tensor_para_size=args.infer_gpu_num,
#             head_num=nemo_model_config["num_attention_heads"],
#             size_per_head=nemo_model_config["hidden_size"] // nemo_model_config["num_attention_heads"],
#             inter_size=nemo_model_config["ffn_hidden_size"],
#             num_layer=nemo_model_config["num_layers"],
#             max_pos_seq_len=nemo_model_config["max_position_embeddings"],
#             weight_data_type=args.weight_data_type,
#             vocab_size=vocab_size,
#             start_id=bos_id,
#             end_id=eos_id,
#         )