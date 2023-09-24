/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/star_coder/StarCoderDecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
StarCoderDecoderLayerWeight<T>::StarCoderDecoderLayerWeight(const int hidden_units, const int inter_size):
    hidden_units_(hidden_units), inter_size_(inter_size)
{
    mallocWeights();
    setWeightPtr();
}

template<typename T>
StarCoderDecoderLayerWeight<T>::~StarCoderDecoderLayerWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < 12; i++) {
            deviceFree(weights_ptr[i]);
        }

        pre_attn_layernorm_weights.gamma = nullptr;
        pre_attn_layernorm_weights.beta = nullptr;
        attention_weights.qkv.kernel = nullptr;
        attention_weights.qkv.bias = nullptr;
        attention_weights.dense_weight.kernel = nullptr;
        attention_weights.dense_weight.bias = nullptr;
        post_attn_layernorm_weights.gamma = nullptr;
        post_attn_layernorm_weights.beta = nullptr;
        mlp_dense_weights.dense_h_to_4h_weight.kernel = nullptr;
        mlp_dense_weights.dense_h_to_4h_weight.bias = nullptr;
        mlp_dense_weights.dense_4h_to_h_weight.kernel = nullptr;
        mlp_dense_weights.dense_4h_to_h_weight.bias = nullptr;
        
        is_maintain_buffer = false;
    }
}

template<typename T>
void StarCoderDecoderLayerWeight<T>::setWeightPtr()
{
    pre_attn_layernorm_weights.gamma = weights_ptr[0];
    pre_attn_layernorm_weights.beta = weights_ptr[1];
    attention_weights.qkv.kernel = weights_ptr[2];
    attention_weights.qkv.bias = weights_ptr[3];
    attention_weights.dense_weight.kernel = weights_ptr[4];
    attention_weights.dense_weight.bias = weights_ptr[5];
    post_attn_layernorm_weights.gamma = weights_ptr[6];
    post_attn_layernorm_weights.beta = weights_ptr[7];

    mlp_dense_weights.dense_h_to_4h_weight.kernel = weights_ptr[8];
    mlp_dense_weights.dense_h_to_4h_weight.bias = weights_ptr[9];
    mlp_dense_weights.dense_4h_to_h_weight.kernel = weights_ptr[10];
    mlp_dense_weights.dense_4h_to_h_weight.bias = weights_ptr[11];

    is_maintain_buffer = true;
}

template<typename T>
void StarCoderDecoderLayerWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], 6144);
    deviceMalloc(&weights_ptr[1], 6144);
    deviceMalloc(&weights_ptr[2], 6144 * 6400);
    deviceMalloc(&weights_ptr[3], 6400);
    deviceMalloc(&weights_ptr[4], 6144 * 6144);
    deviceMalloc(&weights_ptr[5], 6144);

    deviceMalloc(&weights_ptr[6], 6144);
    deviceMalloc(&weights_ptr[7], 6144);
    deviceMalloc(&weights_ptr[8], 24576 * 6144);
    deviceMalloc(&weights_ptr[9], 24576);
    deviceMalloc(&weights_ptr[10], 6144 * 24576);
    deviceMalloc(&weights_ptr[11], 6144);
}

template struct StarCoderDecoderLayerWeight<float>;
template struct StarCoderDecoderLayerWeight<half>;

}  // namespace fastertransformer
