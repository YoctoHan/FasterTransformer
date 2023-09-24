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

#include "src/fastertransformer/models/star_coder/StarCoderWeight.h"

namespace fastertransformer {

template<typename T>
StarCoderWeight<T>::StarCoderWeight(
    const int hidden_units, const int inter_size, const int vocab_size, const int num_layer, const int max_seq_len):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    num_layer_(num_layer),
    max_seq_len_(max_seq_len)
{
    printf("\n============================ -DEBUG- ============================\n");
    printf("StarCoderWeight<T>::StarCoderWeight");
    printf("\n============================= -END- =============================\n");
    for (int l = 0; l < num_layer_; l++) {
        printf("\nprepare %d layer weights\n", l);
        decoder_layer_weights.push_back(new StarCoderDecoderLayerWeight<T>());
    }

    mallocWeights();
    setWeightPtr();
}

template<typename T>
void StarCoderWeight<T>::resizeLayer(const int num_layer)
{
    printf("\n============================ -DEBUG- ============================\n");
    printf("StarCoderWeight::resizeLayer");
    printf("\n============================= -END- =============================\n");
    num_layer_ = num_layer;
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(new StarCoderDecoderLayerWeight<T>());
    }
}

template<typename T>
StarCoderWeight<T>::~StarCoderWeight()
{
    printf("\n============================ -DEBUG- ============================\n");
    printf("StarCoderWeight<T>::~StarCoderWeight");
    printf("\n============================= -END- =============================\n");

    if (is_maintain_buffer == true) {
        for (int i = 0; i < 4; i++) {
            deviceFree(weights_ptr[i]);
        }
        
        position_encodings_table = nullptr;
        word_embeddings_table = nullptr;
        final_layernorm.beta = nullptr;
        final_layernorm.gamma = nullptr;
        
        is_maintain_buffer = false;
    }
}

template<typename T>
void StarCoderWeight<T>::setWeightPtr()
{
    position_encodings_table = weights_ptr[0];
    word_embeddings_table = weights_ptr[1];
    final_layernorm.beta = weights_ptr[2];
    final_layernorm.gamma = weights_ptr[3];
}

template<typename T>
void StarCoderWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], 8192 * 6144);
    deviceMalloc(&weights_ptr[1], 49152 * 6144);
    deviceMalloc(&weights_ptr[2], 6144);
    deviceMalloc(&weights_ptr[3], 6144);

    is_maintain_buffer = true;
}

template struct StarCoderWeight<float>;
template struct StarCoderWeight<half>;

}  // namespace fastertransformer
