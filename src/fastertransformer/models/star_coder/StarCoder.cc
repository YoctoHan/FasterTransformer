/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 * Copyright (c) 2022, SK Telecom Authored by A. Dialog
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

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/StarCoder.cc

#include "src/fastertransformer/models/star_coder/StarCoder.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/macro.h"
#include "src/fastertransformer/models/star_coder/StarCoderWeight.h"
#include "src/fastertransformer/models/star_coder/Request.h"
#include "src/fastertransformer/models/star_coder/star_coder_params.h"
#include "src/fastertransformer/models/star_coder/star_coder_utils.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace fastertransformer {

template<typename T>
StarCoder<T>::StarCoder(size_t                       head_num,
                    size_t                       kv_head_num,
                    size_t                       size_per_head,
                    size_t                       inter_size,
                    size_t                       num_layer,
                    size_t                       vocab_size,
                    const StarCoderAttentionParams&  attn_params,
                    float                        norm_eps,
                    int                          max_batch_size,
                    int                          max_context_token_num,
                    int                          session_len,
                    int                          step_length,
                    int                          start_id,
                    int                          end_id,
                    int                          cache_max_entry_count,
                    int                          cache_chunk_size,
                    int                          quant_policy,
                    bool                         use_context_fmha,
                    StarCoderWeight<T>*              weights,
                    NcclParam                    tensor_para,
                    cudaStream_t                 stream,
                    cublasMMWrapper*             cublas_wrapper,
                    IAllocator*                  allocator,
                    bool                         is_free_buffer_after_forward,
                    cudaDeviceProp*              cuda_device_prop):
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    vocab_size_padded_(vocab_size),
    rmsnorm_eps_(norm_eps),
    start_id_(start_id),
    end_id_(end_id),
    hidden_units_(head_num * size_per_head),
    local_head_num_(head_num / tensor_para.world_size_),
    weights_(weights),
    tensor_para_(tensor_para),
    stream_(stream),
    cublas_wrapper_(cublas_wrapper),
    allocator_(allocator),
    is_free_buffer_after_forward_(is_free_buffer_after_forward),
    cuda_device_prop_(cuda_device_prop),
    debug_(isDebug()),
    step_length_(step_length)

{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    size_t elem_bits = 0;
    if (quant_policy & QuantPolicy::kCacheKVInt8) {
        elem_bits = sizeof(int8_t) * 8;
        if (use_context_fmha) {
            FT_LOG_ERROR("use_context_fmha not support int8");
            assert(0);
        }
    }
    else {
        elem_bits = sizeof(T) * 8;
    }

    const size_t local_kv_head_num = kv_head_num;

    initialize(attn_params, kv_head_num, use_context_fmha, quant_policy);
}

template<typename T>
StarCoder<T>::~StarCoder()
{
    delete decoder_;
    delete dynamic_decode_layer_;
    delete context_decoder_;
}

template<typename T>
void StarCoder<T>::initialize(const StarCoderAttentionParams& attn_params,
                              size_t                          kv_head_num,
                              bool                            use_context_fmha,
                              int                             quant_policy)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    context_decoder_ = new StarCoderContextDecoder<T>(head_num_,
                                                  kv_head_num,
                                                  size_per_head_,
                                                  inter_size_,
                                                  num_layer_,
                                                  attn_params,
                                                  rmsnorm_eps_,
                                                  tensor_para_,
                                                  stream_,
                                                  cublas_wrapper_,
                                                  allocator_,
                                                  is_free_buffer_after_forward_,
                                                  use_context_fmha,
                                                  quant_policy);

    // decoder_ = new StarCoderDecoder<T>(head_num_,
    //                                kv_head_num,
    //                                size_per_head_,
    //                                inter_size_,
    //                                num_layer_,
    //                                attn_params,
    //                                rmsnorm_eps_,
    //                                tensor_para_,
    //                                stream_,
    //                                cublas_wrapper_,
    //                                allocator_,
    //                                is_free_buffer_after_forward_,
    //                                quant_policy);

    // dynamic_decode_layer_ = new DynamicDecodeLayer<float>(vocab_size_,
    //                                                       vocab_size_padded_,
    //                                                       0,            // end_id, deprecated
    //                                                       stream_,
    //                                                       cublas_wrapper_,
    //                                                       allocator_,
    //                                                       is_free_buffer_after_forward_,
    //                                                       cuda_device_prop_);
}

template<typename T>
void StarCoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void StarCoder<T>::allocateBuffer(size_t batch_size,
                                  size_t max_session_len,
                                  size_t memory_len,
                                  size_t max_input_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const size_t batchxbeam       = batch_size;

    context_decoder_input_buf_ =
        (T*)allocator_->reMalloc(context_decoder_input_buf_, sizeof(T) * max_context_token_num_ * hidden_units, false);
    context_decoder_output_buf_ =
        (T*)allocator_->reMalloc(context_decoder_output_buf_, sizeof(T) * max_context_token_num_ * hidden_units, false);
    context_decoder_ids_buf_ =
        (int*)allocator_->reMalloc(context_decoder_ids_buf_, sizeof(int) * max_context_token_num_, false);

    decoder_input_buf_  = (T*)allocator_->reMalloc(decoder_input_buf_, sizeof(T) * batchxbeam * hidden_units, false);
    decoder_output_buf_ = (T*)allocator_->reMalloc(decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units, false);

    input_ids_buf_      = (int*)allocator_->reMalloc(input_ids_buf_, sizeof(int) * batchxbeam * session_len, true);
    input_length_buf_   = (int*)allocator_->reMalloc(input_length_buf_, sizeof(int) * batchxbeam);
    history_length_buf_ = (int*)allocator_->reMalloc(history_length_buf_, sizeof(int) * batchxbeam);
    context_length_buf_ = (int*)allocator_->reMalloc(context_length_buf_, sizeof(int) * batchxbeam);

    total_padding_count_ = (int*)allocator_->reMalloc(total_padding_count_, sizeof(int) * batchxbeam, false);
    sequence_lengths_    = (int*)allocator_->reMalloc(sequence_lengths_, sizeof(int) * batchxbeam, false);

    k_cache_ptr_buf_ = (uint64_t*)allocator_->reMalloc(k_cache_ptr_buf_, sizeof(uint64_t) * batchxbeam);
    v_cache_ptr_buf_ = (uint64_t*)allocator_->reMalloc(v_cache_ptr_buf_, sizeof(uint64_t) * batchxbeam);

    logits_buf_       = (float*)allocator_->reMalloc(logits_buf_, sizeof(float) * batchxbeam * vocab_size, false);
    local_logits_buf_ = (float*)allocator_->reMalloc(local_logits_buf_, sizeof(float) * batchxbeam * vocab_size, false);

    token_ids_buf_ = (int*)allocator_->reMalloc(token_ids_buf_, sizeof(int) * batchxbeam * session_len * 2, true);

    end_ids_buf_   = (int*)allocator_->reMalloc(end_ids_buf_, sizeof(int) * batch_size, false);
    finished_buf_  = (bool*)allocator_->reMalloc(finished_buf_, sizeof(bool) * batchxbeam, false);
    seq_limit_len_ = (uint32_t*)allocator_->reMalloc(seq_limit_len_, sizeof(uint32_t) * batch_size, false);

    is_allocate_buffer_ = true;
}

template<typename T>
void StarCoderBatch<T>::allocatePersistantBuffer(size_t max_batch_size)
{
    output_ids_buf_ = (int*)allocator_->reMalloc(output_ids_buf_, sizeof(int) * max_batch_size * session_len_, true);

    stop_words_buf_ =
        (int*)allocator_->reMalloc(stop_words_buf_, sizeof(int) * max_batch_size * kMaxStopBadWordsLen, true);
    bad_words_buf_ =
        (int*)allocator_->reMalloc(bad_words_buf_, sizeof(int) * max_batch_size * kMaxStopBadWordsLen, true);

    h_runtime_top_k_ = (int*)allocator_->reMalloc(h_runtime_top_k_, sizeof(int) * max_batch_size, true, true);
    h_runtime_top_p_ = (float*)allocator_->reMalloc(h_runtime_top_p_, sizeof(float) * max_batch_size, true, true);
    h_temperature_   = (float*)allocator_->reMalloc(h_temperature_, sizeof(float) * max_batch_size, true, true);
    h_repetition_penalty_ =
        (float*)allocator_->reMalloc(h_repetition_penalty_, sizeof(float) * max_batch_size, true, true);
    h_random_seed_ = (uint64_t*)allocator_->reMalloc(h_random_seed_, sizeof(uint64_t) * max_batch_size, true, true);

    sampling_params_ = {{"stop_words_list", stop_words_buf_},
                        {"bad_words_list", bad_words_buf_},
                        {"runtime_top_k", h_runtime_top_k_},
                        {"runtime_top_p", h_runtime_top_p_},
                        {"temperature", h_temperature_},
                        {"repetition_penalty", h_repetition_penalty_},
                        {"random_seed", h_random_seed_}};

    topk_curandstate_buf_ = allocator_->reMalloc(topk_curandstate_buf_, sizeof(curandState_t) * max_batch_size, true);
    topp_curandstate_buf_ = allocator_->reMalloc(topp_curandstate_buf_, sizeof(curandState_t) * max_batch_size, true);

    {
        h_input_ids_buf_ =
            (int*)allocator_->reMalloc(h_input_ids_buf_, sizeof(int) * max_batch_size * session_len_, false, true);
        h_input_length_buf_ =
            (int*)allocator_->reMalloc(h_input_length_buf_, sizeof(int) * max_batch_size, false, true);
        h_history_length_buf_ =
            (int*)allocator_->reMalloc(h_history_length_buf_, sizeof(int) * max_batch_size, false, true);
        h_context_length_buf_ =
            (int*)allocator_->reMalloc(h_context_length_buf_, sizeof(int) * max_batch_size, false, true);
        h_sequence_lengths_ =
            (int*)allocator_->reMalloc(h_sequence_lengths_, sizeof(int) * max_batch_size, false, true);
        h_k_cache_ptr_buf_ =
            (uintptr_t*)allocator_->reMalloc(h_k_cache_ptr_buf_, sizeof(uintptr_t) * max_batch_size, true, true);
        h_v_cache_ptr_buf_ =
            (uintptr_t*)allocator_->reMalloc(h_v_cache_ptr_buf_, sizeof(uintptr_t) * max_batch_size, true, true);
        h_finished_buf_ = (bool*)allocator_->reMalloc(h_finished_buf_, sizeof(bool) * max_batch_size, false, true);
        h_seq_limit_len_ =
            (uint32_t*)allocator_->reMalloc(h_seq_limit_len_, sizeof(uint32_t) * max_batch_size, false, true);
    }

    is_allocate_persistant_buffer_ = true;
}

template<typename T>
void LlamaBatch<T>::freeBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)&context_decoder_input_buf_);
        allocator_->free((void**)&context_decoder_output_buf_);
        allocator_->free((void**)&context_decoder_ids_buf_);

        allocator_->free((void**)&decoder_input_buf_);
        allocator_->free((void**)&decoder_output_buf_);

        allocator_->free((void**)&input_ids_buf_);
        allocator_->free((void**)&input_length_buf_);
        allocator_->free((void**)&history_length_buf_);
        allocator_->free((void**)&context_length_buf_);

        allocator_->free((void**)&total_padding_count_);
        allocator_->free((void**)&sequence_lengths_);

        allocator_->free((void**)&k_cache_ptr_buf_);
        allocator_->free((void**)&v_cache_ptr_buf_);

        allocator_->free((void**)&logits_buf_);
        allocator_->free((void**)&local_logits_buf_);

        if (local_context_logits_buf_) {
            allocator_->free((void**)&local_context_logits_buf_);
        }
        if (context_logits_buf_) {
            allocator_->free((void**)&context_logits_buf_);
        }

        allocator_->free((void**)&token_ids_buf_);

        allocator_->free((void**)&end_ids_buf_);
        allocator_->free((void**)&finished_buf_);
        allocator_->free((void**)&seq_limit_len_);

        is_allocate_buffer_ = false;
    }

    if (is_allocate_persistant_buffer_) {
        allocator_->free((void**)&h_input_ids_buf_, true);
        allocator_->free((void**)&h_input_length_buf_, true);
        allocator_->free((void**)&h_history_length_buf_, true);
        allocator_->free((void**)&h_context_length_buf_, true);
        allocator_->free((void**)&h_sequence_lengths_, true);
        allocator_->free((void**)&h_k_cache_ptr_buf_, true);
        allocator_->free((void**)&h_v_cache_ptr_buf_, true);
        allocator_->free((void**)&h_seq_limit_len_, true);
        allocator_->free((void**)&h_finished_buf_, true);

        allocator_->free((void**)&output_ids_buf_);

        is_allocate_persistant_buffer_ = false;
    }
}

template<typename T>
void StarCoder<T>::forward(std::unordered_map<std::string, Tensor>*       outputs,
                           const std::unordered_map<std::string, Tensor>* inputs)
{
    // input_tensors:
    //      input_ids [batch_size * beam, max_input_length]
    //      input_lengths [batch_size * beam]
    //      max_output_seq_len [1] on cpu

    // output_tensors:
    //      output_ids [batch_size, beam, max_output_seq_len]
    //      sequence_length [batch_size * beam]

    // Step is from max_input_length ~ max_output_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.
    // When there is no input_ids, put the start token at step 0 of output_ids_buf_. After forward, only copy
    // the step 1 ~ max_output_seq_len of output_ids_buf_ to output_tensors->at(0).data

    allocateBuffer();

    int max_input_length = input_tensors->at("input_ids").shape[1];
    const int* input_length_ptr = (const int*)(input_tensors->at("input_lengths").data);
    const size_t max_output_seq_len = (size_t)(*(int*)input_tensors->at("max_output_seq_len").data)
                                      + (max_input_length == 0 ? 1 : 0);  // additional 1 to put start token

    const size_t batch_size = output_tensors->at("output_ids").shape[0];
    int* sequence_lengths = (int*)(output_tensors->at("sequence_length").data);
    const DataType data_type = getTensorType<T>();

    // initialize the output ids
    cudaMemsetAsync(output_ids_buf_, 0, sizeof(int) * batch_size * max_seq_len_, stream_);

    // handle first step
    const std::vector<size_t> self_k_cache_size = {num_layer_,
                                                   batch_size,
                                                   head_num_,
                                                   size_per_head_ / (16 / sizeof(T)),
                                                   max_output_seq_len,
                                                   16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_size = {
        num_layer_, batch_size, head_num_, max_output_seq_len, size_per_head_};

    invokeBuildDecoderAttentionMask(
        input_attention_mask_, input_length_ptr, batch_size, max_input_length, stream_);
    sync_check_cuda_error();

    invokeInputIdsEmbeddingLookupPosEncoding(context_decoder_input_buf_,
                                             output_ids_buf_,
                                             gpt_weights->pre_decoder_embedding_table,
                                             gpt_weights->position_encoding_table,
                                             (int*)input_tensors->at(0).data,
                                             1,
                                             max_input_length,
                                             max_input_length,
                                             batch_size,
                                             hidden_units_,
                                             stream_);
    sync_check_cuda_error();

    std::vector<Tensor> decoder_input_tensors{
        Tensor{MEMORY_GPU,
                data_type,
                {batch_size, (size_t)max_input_length, hidden_units_},
                context_decoder_input_buf_},
        Tensor{MEMORY_GPU,
                data_type,
                {batch_size , 1, (size_t)max_input_length, (size_t)max_input_length},
                input_attention_mask_}};

    const int src_cache_id = 0;
    std::vector<Tensor> decoder_output_tensors{
        Tensor{MEMORY_GPU,
                data_type,
                {batch_size, (size_t)max_input_length, hidden_units_},
                context_decoder_output_buf_},
        Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_caches_[src_cache_id]},
        Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_caches_[src_cache_id]}};

    gpt_context_decoder_->forward(
        &decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);
}

template class StarCoder<half>;
template class StarCoder<float>;

}  // namespace fastertransformer
