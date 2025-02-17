/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "src/fastertransformer/utils/cublasAlgoMap.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/gemm_test/gemm_func.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#ifdef ENABLE_BF16
#include <cuda_fp16.h>
#endif
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#include <cuda_profiler_api.h>
#include <map>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

namespace fastertransformer {

template<typename T>
void generate_star_coder_gemm_config(int   batch_size,
                              int   beam_width,
                              int   seq_len,
                              int   head_num,
                              int   size_per_head,
                              int   inter_size,
                              int   vocab_size,
                              int   tensor_para_size,
                              void* buffer_in,
                              bool  isAppend);

size_t calStarCoderGemmTestBufSizeInByte(size_t         batch_size,
                                   size_t         beam_width,
                                   size_t         max_input_len,
                                   size_t         head_num,
                                   size_t         size_per_head,
                                   size_t         inter_size,
                                   size_t         vocab_size,
                                   size_t         tensor_para_size,
                                   CublasDataType data_type);

}  // namespace fastertransformer
