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

#pragma once

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/models/star_coder/StarCoderDecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct StarCoderWeight {

    StarCoderWeight() = default;
    StarCoderWeight(const int hidden_units,
                    const int inter_size,
                    const int vocab_size,
                    const int num_layer,
                    const int max_seq_len);
    ~StarCoderWeight();
    StarCoderWeight(const StarCoderWeight& other);
    StarCoderWeight& operator=(const StarCoderWeight& other);

    void resizeLayer(const int num_layer);

    std::vector<StarCoderDecoderLayerWeight<T>*> decoder_layer_weights;
    const T* position_encodings_table = nullptr;
    const T* word_embeddings_table = nullptr;
    LayerNormWeight<T> final_layernorm;

private:
    void setWeightPtr();
    void mallocWeights();

    int hidden_units_;
    int inter_size_;
    int vocab_size_;
    int num_layer_;
    int max_seq_len_;
    bool is_maintain_buffer = false;
    T* weights_ptr[4];
};

}  // namespace fastertransformer
