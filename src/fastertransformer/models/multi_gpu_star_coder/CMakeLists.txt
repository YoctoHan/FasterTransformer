# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_library(ParallelStarCoderDecoderLayerWeight STATIC ParallelStarCoderDecoderLayerWeight.cc)
set_property(TARGET ParallelStarCoderDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelStarCoderDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelStarCoderDecoderLayerWeight PUBLIC memory_utils calibrate_quantize_weight_kernels transpose_int8_kernels cuda_utils logger)

add_library(ParallelStarCoderWeight STATIC ParallelStarCoderWeight.cc)
set_property(TARGET ParallelStarCoderWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelStarCoderWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelStarCoderWeight PUBLIC ParallelStarCoderDecoderLayerWeight cuda_utils logger)

add_library(ParallelStarCoderContextDecoder STATIC ParallelStarCoderContextDecoder.cc)
set_property(TARGET ParallelStarCoderContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelStarCoderContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelStarCoderContextDecoder PUBLIC -lcudart TensorParallelGeluFfnLayer TensorParallelReluFfnLayer
                                                TensorParallelStarCoderContextAttentionLayer layernorm_kernels
                                                add_residual_kernels bert_preprocess_kernels nccl_utils star_coder_kernels tensor
                                                nvtx_utils cuda_utils logger)

add_library(ParallelStarCoderDecoder STATIC ParallelStarCoderDecoder.cc)
set_property(TARGET ParallelStarCoderDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelStarCoderDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelStarCoderDecoder PUBLIC -lcudart TensorParallelGeluFfnLayer TensorParallelReluFfnLayer
                                                TensorParallelDecoderSelfAttentionLayer layernorm_kernels
                                                add_residual_kernels nccl_utils tensor cuda_utils logger)

add_library(ParallelStarCoder STATIC ParallelStarCoder.cc)
set_property(TARGET ParallelStarCoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelStarCoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelStarCoder PUBLIC -lcudart ParallelStarCoderDecoder ParallelStarCoderContextDecoder decoding_kernels star_coder_kernels
                      DynamicDecodeLayer BaseBeamSearchLayer bert_preprocess_kernels gen_relative_pos_bias ParallelStarCoderWeight
                      custom_ar_comm logprob_kernels cuda_utils logger nvtx_utils)

add_executable(star_coder_gemm star_coder_gemm.cc)
target_link_libraries(star_coder_gemm PUBLIC -lcudart star_coder_gemm_func memory_utils cuda_utils logger)
