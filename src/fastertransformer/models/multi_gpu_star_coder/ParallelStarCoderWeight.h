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

#include "src/fastertransformer/kernels/layernorm_kernels.h"

#include "src/fastertransformer/models/multi_gpu_star_coder/ParallelStarCoderDecoder.h"
#include "src/fastertransformer/models/multi_gpu_star_coder/ParallelStarCoderDecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/prompt_learning.h"

namespace fastertransformer {

template<typename T>
struct ParallelStarCoderWeight {

    ParallelStarCoderWeight() = default;
    ParallelStarCoderWeight(const int                                  hidden_units,
                            const int                                  inter_size,
                            const int                                  vocab_size,
                            const int                                  num_layer,
                            const int                                  max_seq_len,
                            const int                                  tensor_para_size,
                            const int                                  tensor_para_rank,
                            const int                                  layer_para_size,
                            const int                                  layer_para_rank,
                            const int                                  int8_mode            = 0,
                            PromptLearningType                         prompt_learning_type = PromptLearningType::no_prompt,
                            std::map<std::string, std::pair<int, int>> prompt_learning_pair = {},
                            starCoderVariantParams                     star_coder_variant_params   = {});
    ~ParallelStarCoderWeight();
    ParallelStarCoderWeight(const ParallelStarCoderWeight& other);
    ParallelStarCoderWeight& operator=(const ParallelStarCoderWeight& other);
    void               loadModel(std::string dir_path);
    void               resizeLayer(const int num_layer, const int int8_mode = 0);

    std::vector<ParallelStarCoderDecoderLayerWeight<T>*> decoder_layer_weights;
    const T*                                             position_encoding_table         = nullptr;
    const T*                                             pre_word_embeddings             = nullptr;
    const T*                                             post_word_embeddings            = nullptr;
    LayerNormWeight<T>                                   post_decoder_layernorm;
    /*
       prompt_learning_pair = vectors of [weight ptr, prompt length] pair
       prompt_length is stored here for compatible prompt learning table
       prefix_prompt weights store as shape [num_layers, 2, num_heads, perfix_seq_len, size_per_head]
       p/prompt tuning weights store as shape [prompt_len, hidden_units]
       idx is the task_name_id of the prompt tables
    */
    std::vector<std::pair<const T*, int>> prompt_learning_table = {};
    inline size_t                         getMaxSeqLen() const
    {
        return max_seq_len_;
    }
    inline void setMaxSeqLen(size_t max_seq_len)
    {
        max_seq_len_ = max_seq_len;
    }

private:
    void setWeightPtr();
    void mallocWeights();
    bool isValidLayerParallelId(int l);

    size_t          hidden_units_;
    size_t            inter_size_;
    size_t            vocab_size_;
    size_t             num_layer_;
    size_t           max_seq_len_;
    size_t      tensor_para_size_;
    size_t      tensor_para_rank_;
    size_t       layer_para_size_;
    size_t       layer_para_rank_;
    size_t int8_mode_    =      0;
    bool   shared_embed_ =  false;

    // prompt learning pair (task_name, (task_name_id, prompt_len))
    PromptLearningType                                        prompt_learning_type_;
    std::map<std::string, std::pair<int, int>>                prompt_learning_pair_;
    bool                                        malloc_load_prompt_weights_ = false;
    // each prompt token's weight size
    size_t prompt_token_weight_size_ =     0;

    bool is_maintain_buffer          = false;

    // The number of base weights of the StarCoder model,
    // positional encoding or pre decoder layernorm can be nullptr.
    //  - 1 for the input layernorm.
    //  - 2 for the attention.
    //  - 1 for the attention dense.
    //  - 1 for the post attention layernorm.
    //  - 1 for the dense h to 4h.
    //  - 1 for the dense 4h to h.
    size_t num_base_weights     =                                 7;
    // weight pointers of length num_base_weights.
    std::vector<T*> weights_ptr = std::vector<T*>(num_base_weights);
};

}  // namespace fastertransformer
