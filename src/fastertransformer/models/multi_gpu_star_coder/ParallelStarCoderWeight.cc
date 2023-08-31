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

#include "src/fastertransformer/models/multi_gpu_star_coder/ParallelStarCoderWeight.h"

namespace fastertransformer {

template<typename T>
ParallelStarCoderWeight<T>::ParallelStarCoderWeight(const int                                  hidden_units,
                                                    const int                                  inter_size,
                                                    const int                                  vocab_size,
                                                    const int                                  num_layer,
                                                    const int                                  max_seq_len,
                                                    const int                                  tensor_para_size,
                                                    const int                                  tensor_para_rank,
                                                    const int                                  layer_para_size,
                                                    const int                                  layer_para_rank,
                                                    const int                                  int8_mode,
                                                    PromptLearningType                         prompt_learning_type,
                                                    std::map<std::string, std::pair<int, int>> prompt_learning_pair,
                                                    starCoderVariantParams                     star_coder_variant_params):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    num_layer_(num_layer),
    max_seq_len_(max_seq_len),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    layer_para_size_(layer_para_size),
    layer_para_rank_(layer_para_rank),
    int8_mode_(int8_mode),
    prompt_learning_type_(prompt_learning_type),
    prompt_learning_pair_(prompt_learning_pair)
{

    printf("\n============================ -DEBUG- ============================\n");
    printf("ParallelStarCoderWeight::ParallelStarCoderWeight");
    printf("\n============================= -END- =============================\n");

    FT_CHECK(num_layer_ % layer_para_size_ == 0);
    // set prompt weight size
    if (prompt_learning_type_ == PromptLearningType::prefix_prompt) {
        prompt_token_weight_size_ = 2 * num_layer_ * hidden_units_ / tensor_para_size_;
    }
    else if (prompt_learning_type_ == PromptLearningType::p_prompt_tuning) {
        prompt_token_weight_size_ = hidden_units_;
    }

    // set if load and malloc prompt weights
    malloc_load_prompt_weights_ = !prompt_learning_pair_.empty()
                                  && (prompt_learning_type_ == PromptLearningType::p_prompt_tuning
                                      || prompt_learning_type_ == PromptLearningType::prefix_prompt);
    decoder_layer_weights.clear();
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            decoder_layer_weights.push_back(new ParallelStarCoderDecoderLayerWeight<T>(
                hidden_units_, inter_size_, tensor_para_size_, tensor_para_rank_, int8_mode_));
        }
        else {
            // Don't malloc and load these layers since we don't use them.
            decoder_layer_weights.push_back(new ParallelStarCoderDecoderLayerWeight<T>());
        }
    }

    printf("\n============================ -DEBUG- ============================\n");
    printf("\n%d\n", isValidLayerParallelId(0));
    printf("\n============================= -END- =============================\n");
    
    mallocWeights();
    setWeightPtr();
}

template<typename T>
ParallelStarCoderWeight<T>::~ParallelStarCoderWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < weights_ptr.size(); i++) {
            if (i == 6 && shared_embed_ && weights_ptr[i] == nullptr) {
                continue;
            }
            deviceFree(weights_ptr[i]);
        }


        position_encoding_table     = nullptr;
        pre_word_embeddings             = nullptr;
        post_word_embeddings             = nullptr;
        post_decoder_layernorm.gamma = nullptr;
        post_decoder_layernorm.beta = nullptr;
        is_maintain_buffer            = false;
    }

    for (int i = 0; i < num_layer_; i++) {
        delete decoder_layer_weights[i];
    }
}

template<typename T>
void ParallelStarCoderWeight<T>::setWeightPtr()
{
    // prompt_learning_table.resize(prompt_learning_pair_.size());

    pre_word_embeddings  = weights_ptr[1];
    // if (shared_embed_ && weights_ptr[6] != weights_ptr[1]) {
    //     deviceFree(weights_ptr[6]);
    //     weights_ptr[6]                = nullptr;
    //     post_word_embeddings.kernel = weights_ptr[1];
    // }
    // else {
    //     post_word_embeddings.kernel = weights_ptr[6];
    // }
    // post_word_embeddings.bias = nullptr;
}

template<typename T>
void ParallelStarCoderWeight<T>::mallocWeights()
{
    weights_ptr.resize(num_base_weights + prompt_learning_pair_.size());

    // word embedding table.
    deviceMalloc(&weights_ptr[1], vocab_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[6], hidden_units_ * vocab_size_);

    // prompt learning tables: malloc weights
    if (malloc_load_prompt_weights_) {
        for (auto const& prompt : prompt_learning_pair_) {
            int    task_name_id   = prompt.second.first;
            int    prompt_length  = prompt.second.second;
            size_t task_weight_id = num_base_weights + (size_t)task_name_id;

            // malloc weights
            T* prompt_weights_ptr = nullptr;
            deviceMalloc(&prompt_weights_ptr, prompt_length * prompt_token_weight_size_);
            weights_ptr[task_weight_id] = prompt_weights_ptr;
        }
    }
    is_maintain_buffer = true;
}

template<typename T>
void ParallelStarCoderWeight<T>::loadModel(std::string dir_path)
{
    FtCudaDataType model_file_type = getModelFileType(dir_path + "/config.ini", "star_coder");
    FT_CHECK(is_maintain_buffer == true);
    loadWeightFromBin<T>(weights_ptr[1], {vocab_size_ * hidden_units_}, dir_path + "/model.wte.bin", model_file_type);
    if (checkIfFileExist(dir_path + "/model.lm_head.weight.bin")) {
        shared_embed_ = false;
        loadWeightFromBin<T>(
            weights_ptr[6], {vocab_size_ * hidden_units_}, dir_path + "/model.lm_head.weight.bin", model_file_type);
    }
    else {
        shared_embed_ = true;
        loadWeightFromBin<T>(
            weights_ptr[6], {vocab_size_ * hidden_units_}, dir_path + "/model.wte.bin", model_file_type);
    }

    // prompt table: load weights from bin
    if (malloc_load_prompt_weights_) {
        for (auto const& prompt : prompt_learning_pair_) {
            std::string task_name      = prompt.first;
            int         task_name_id   = prompt.second.first;
            int         prompt_length  = prompt.second.second;
            size_t      task_weight_id = num_base_weights + (size_t)task_name_id;

            std::string prompt_weight_path_name = (prompt_learning_type_ == PromptLearningType::p_prompt_tuning) ?
                                                      (dir_path + "/model.prompt_table." + task_name + ".weight.bin") :
                                                      (dir_path + "/model.prefix_prompt." + task_name + ".weight."
                                                       + std::to_string(tensor_para_rank_) + ".bin");

            if (prompt_length > 0) {
                loadWeightFromBin<T>(weights_ptr[task_weight_id],
                                     {prompt_length * prompt_token_weight_size_},
                                     prompt_weight_path_name,
                                     model_file_type);
            }
        }
    }
    setWeightPtr();

    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            decoder_layer_weights[l]->loadModel(dir_path + "/model.layers." + std::to_string(l), model_file_type);
        }
    }
}

template<typename T>
bool ParallelStarCoderWeight<T>::isValidLayerParallelId(int l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / layer_para_size_));
    return l < num_layer_ && (l >= local_num_layer * layer_para_rank_)
           && (l < local_num_layer * (layer_para_rank_ + 1));
}

template<typename T>
void ParallelStarCoderWeight<T>::resizeLayer(const int num_layer, const int int8_mode)
{

    printf("\n============================ -DEBUG- ============================\n");
    printf("ParallelStarCoderWeight::resizeLayer");
    printf("\n============================= -END- =============================\n");
    
    int8_mode_ = int8_mode;
    num_layer_ = num_layer;
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(new ParallelStarCoderDecoderLayerWeight<T>(int8_mode_));
    }
}

template struct ParallelStarCoderWeight<float>;
template struct ParallelStarCoderWeight<half>;
#ifdef ENABLE_BF16
template struct ParallelStarCoderWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
