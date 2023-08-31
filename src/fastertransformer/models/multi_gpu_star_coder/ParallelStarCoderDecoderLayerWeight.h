/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <string>

#include "src/fastertransformer/kernels/calibrate_quantize_weight_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"

namespace fastertransformer {

// The maximum value of sequence length, a random number larger than
// the maximum length by our kernel support length (4096). This value
// should be meaningful only for no positional encoding case.
#define FT_SEQ_LEN_MAX (16384)

struct starCoderVariantParams {
    // StarCoder default params
    float          layernorm_eps   = 1e-6f;
    LayerNormType  layernorm_type  = LayerNormType::pre_layernorm;
    ActivationType activation_type = ActivationType::Gelu;
    // Whether to have a learnt positional encoding.
    bool has_positional_encoding = true;
    // A layernom just after the word embedding and before the decoder.
    bool has_pre_decoder_layernorm = false;
    // A layernom after the decoder.
    bool has_post_decoder_layernorm = true;
    // detoxification adapters. refer to
    bool   has_adapters       = false;
    size_t adapter_inter_size = 0;
    // Whether to use the attention linear positional bias
    bool use_attention_linear_bias = false;
};

template<typename T>
struct ParallelStarCoderDecoderLayerWeight {
public:
    ParallelStarCoderDecoderLayerWeight() = default;
    ParallelStarCoderDecoderLayerWeight(const int int8_mode);
    ParallelStarCoderDecoderLayerWeight(const int        hidden_units,
                                  const int        inter_size,
                                  const int        tensor_para_size,
                                  const int        tensor_para_rank,
                                  const int        int8_mode          = 0,
                                  starCoderVariantParams star_coder_variant_params = {});
    ~ParallelStarCoderDecoderLayerWeight();
    ParallelStarCoderDecoderLayerWeight(const ParallelStarCoderDecoderLayerWeight& other);
    ParallelStarCoderDecoderLayerWeight& operator=(const ParallelStarCoderDecoderLayerWeight& other);
    void                           loadModel(std::string dir_path, FtCudaDataType model_file_type);
#ifdef SPARSITY_ENABLED
    void compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim);
#endif
    void transposeCalibrateQuantizeWeight();
    void transposeWeight();

    LayerNormWeight<T>     pre_attn_layernorm_weights;
    MQAWeight<T>                    attention_weights;
    LayerNormWeight<T>    post_attn_layernorm_weights;
    StarCoderFfnWeight<T>           mlp_dense_weights;

private:
    void copyFrom(const ParallelStarCoderDecoderLayerWeight& other);
    void setWeightPtr();
    void mallocWeights();

protected:
    size_t hidden_units_;
    size_t inter_size_;
    size_t tensor_para_size_  = 1;
    size_t tensor_para_rank_  = 0;
    bool   is_maintain_buffer = false;
    int    int8_mode_         = 0;

    // gpt varians params. e.g. detoxification adapters
    starCoderVariantParams star_coder_variant_params_;

    std::vector<T*> weights_ptr = std::vector<T*>(20, nullptr);

    std::vector<int8_t*> int8_weights_ptr      = std::vector<int8_t*>(8, nullptr);
    std::vector<T*>      weight_only_scale_ptr = std::vector<T*>(8, nullptr);

    std::vector<float*> scale_ptr       = std::vector<float*>(8, nullptr);
    std::vector<float*> scale_out_ptr   = std::vector<float*>(8, nullptr);
    std::vector<float*> scale_inter_ptr = std::vector<float*>(8, nullptr);
    cudaStream_t        stream_         = 0;

#ifdef SPARSITY_ENABLED
    std::vector<T*> sp_weights_ptr        = std::vector<T*>(8, nullptr);
    bool            is_maintain_sp_buffer = false;
#endif
};

}  // namespace fastertransformer
