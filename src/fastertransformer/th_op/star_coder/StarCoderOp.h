/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "src/fastertransformer/models/multi_gpu_star_coder/ParallelStarCoder.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

using std::vector;

class IFStarCoder {
public:
    virtual ~IFStarCoder() {}
    virtual void forward(th::Tensor&                                   input_ids,
                         th::Tensor&                               input_lengths,
                         th::Tensor&                                  output_ids,
                         th::Tensor&                            sequence_lengths,
                         th::Tensor&                               cum_log_probs,
                         const size_t                         request_output_len,
                         const size_t                                 beam_width,
                         th::optional<th::Tensor>                      top_k_opt,
                         th::optional<th::Tensor>                      top_p_opt,
                         th::optional<th::Tensor> beam_search_diversity_rate_opt,
                         th::optional<th::Tensor>                temperature_opt,
                         th::optional<th::Tensor>                len_penalty_opt,
                         th::optional<th::Tensor>         repetition_penalty_opt,
                         th::optional<th::Tensor>                random_seed_opt,
                         th::optional<int64_t>          return_cum_log_probs_opt) = 0;
};

template<typename T>
class FTStarCoder: public IFStarCoder {
public:
    FTStarCoder(const size_t                                      head_num,
                const size_t                                 size_per_head,
                const size_t                                    inter_size,
                const size_t                                     layer_num,
                const size_t                                    vocab_size,
                const ft::starCoderVariantParams star_coder_variant_params,
                const int                                         start_id,
                const int                                           end_id,
                const bool                                          sparse,
                const vector<th::Tensor>                           weights):
        head_num_(head_num),
        size_per_head_(size_per_head),
        inter_size_(inter_size),
        layer_num_(layer_num),
        vocab_size_(vocab_size),
        star_coder_variant_params_(star_coder_variant_params),
        start_id_(start_id),
        end_id_(end_id),
        sparse_(sparse),
        weights_(weights) 
    {

        printf("\n============================ -DEBUG- ============================\n");
        printf("FTStarCoder::FTStarCoder");
        printf("\n============================= -END- =============================\n");

        ft::check_cuda_error(cublasLtCreate(&cublasltHandle_));
        
        std::string sp_config_fname = "";
        cublas_algo_map_            = new ft::cublasAlgoMap(GEMM_CONFIG, sp_config_fname);
        cublas_wrapper_mutex_       = new std::mutex();

        star_coder_weights_.resizeLayer(layer_num_);
        for (int i = 0; i < (int)layer_num_; i++) {
            star_coder_weights_.decoder_layer_weights[i]->pre_attn_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 0 * layer_num_]);
            star_coder_weights_.decoder_layer_weights[i]->pre_attn_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 1 * layer_num_]);
            star_coder_weights_.decoder_layer_weights[i]->attention_weights.query_weight.kernel =
                get_ptr<T>(weights_[i + 2 * layer_num_]);
            star_coder_weights_.decoder_layer_weights[i]->attention_weights.query_weight.bias =
                get_ptr<T>(weights_[i + 3 * layer_num_]);
            star_coder_weights_.decoder_layer_weights[i]->attention_weights.key_value_weight.kernel =
                get_ptr<T>(weights_[i + 4 * layer_num_]);
            star_coder_weights_.decoder_layer_weights[i]->attention_weights.key_value_weight.bias =
                get_ptr<T>(weights_[i + 5 * layer_num_]);
            star_coder_weights_.decoder_layer_weights[i]->attention_weights.dense_weight.kernel =
                get_ptr<T>(weights_[i + 6 * layer_num_]);
            star_coder_weights_.decoder_layer_weights[i]->attention_weights.dense_weight.bias =
                get_ptr<T>(weights_[i + 7 * layer_num_]);
            star_coder_weights_.decoder_layer_weights[i]->post_attn_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 8 * layer_num_]);
            star_coder_weights_.decoder_layer_weights[i]->post_attn_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 9 * layer_num_]);
            star_coder_weights_.decoder_layer_weights[i]->mlp_dense_weights.dense_h_to_4h_weight.kernel =
                get_ptr<T>(weights_[i + 10 * layer_num_]);
            star_coder_weights_.decoder_layer_weights[i]->mlp_dense_weights.dense_h_to_4h_weight.bias =
                get_ptr<T>(weights_[i + 11 * layer_num_]);
            star_coder_weights_.decoder_layer_weights[i]->mlp_dense_weights.dense_4h_to_h_weight.kernel =
                get_ptr<T>(weights_[i + 12 * layer_num_]);
            star_coder_weights_.decoder_layer_weights[i]->mlp_dense_weights.dense_4h_to_h_weight.bias =
                get_ptr<T>(weights_[i + 13 * layer_num_]);
        }

        star_coder_weights_.post_decoder_layernorm.gamma   = get_ptr<T>(weights_[14 * layer_num_ + 0]);
        star_coder_weights_.post_decoder_layernorm.beta    = get_ptr<T>(weights_[14 * layer_num_ + 1]);
        star_coder_weights_.pre_word_embeddings             = get_ptr<T>(weights_[14 * layer_num_ + 2]);
        star_coder_weights_.post_word_embeddings            = get_ptr<T>(weights_[14 * layer_num_ + 2]);
        star_coder_weights_.position_encoding_table        = get_ptr<T>(weights_[14 * layer_num_ + 3]);

        ft::check_cuda_error(cudaGetDeviceProperties(&prop_, 0));
    }

    ~FTStarCoder() override
    {
        cublasLtDestroy(cublasltHandle_);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void forward(th::Tensor&                                   input_ids,
                 th::Tensor&                               input_lengths,
                 th::Tensor&                                  output_ids,
                 th::Tensor&                            sequence_lengths,
                 th::Tensor&                               cum_log_probs,
                 const size_t                         request_output_len,
                 const size_t                                 beam_width,
                 th::optional<th::Tensor>                      top_k_opt,
                 th::optional<th::Tensor>                      top_p_opt,
                 th::optional<th::Tensor> beam_search_diversity_rate_opt,
                 th::optional<th::Tensor>                temperature_opt,
                 th::optional<th::Tensor>                len_penalty_opt,
                 th::optional<th::Tensor>         repetition_penalty_opt,
                 th::optional<th::Tensor>                random_seed_opt,
                 th::optional<int64_t>          return_cum_log_probs_opt) override
    {

        printf("\n============================ -DEBUG- ============================\n");
        printf("FTStarCoder::forward");
        printf("\n============================= -END- =============================\n");

        int  return_cum_log_probs   = return_cum_log_probs_opt.has_value() ? (int)return_cum_log_probs_opt.value() : 0;
        auto stream                 =                                        at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t cublasHandle =                                             at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublasHandle, stream);
        ft::Allocator<ft::AllocatorType::TH> allocator      =                   ft::Allocator<ft::AllocatorType::TH>();
        ft::cublasMMWrapper                  cublas_wrapper = ft::cublasMMWrapper(
                           cublasHandle, cublasltHandle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, &allocator);

        if (std::is_same<T, half>::value) {
            cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
        }
#ifdef ENABLE_BF16
        else if (std::is_same<T, __nv_bfloat16>::value) {
            cublas_wrapper.setBF16GemmConfig();
        }
#endif
        else if (std::is_same<T, float>::value) {
            cublas_wrapper.setFP32GemmConfig();
        }

        const size_t request_batch_size =                    (size_t)input_ids.size(0);
        const size_t max_input_length   =                    (size_t)input_ids.size(1);
        const int    total_output_len   = (int)(max_input_length + request_output_len);

        ft::NcclParam tensor_para;
        ft::NcclParam pipeline_para;

        ft::AttentionType attention_type =
            ft::getAttentionType<T>(size_per_head_,
                                    ft::getSMVersion(),
                                    true,
                                    max_input_length,  // gpt supports any-seq-length fmha
                                    true,              // is_fuse
                                    false,             // with_relative_position_bias
                                    true);             // causal_mask

        size_t kv_head_num = 1;
        StarCoderAttentionParams  attn_params;
        int max_batch_size = 32; 
        int max_contex_token_num = 8192;
        int session_len = 2048;
        int step_length = 1;
        int cache_max_entry_count = 48;
        int cache_chunk_size = 1;
        int quant_policy = 0;
        bool use_context_fmha = true;
        bool is_free_buffer_after_forward = false;

        ft::StarCoder<T> star_coder = ft::StarCoder<T>(head_num_,
                                                       kv_head_num,
                                                       size_per_head_,
                                                       inter_size_,
                                                       layer_num_,
                                                       vocab_size_,
                                                       attn_params,
                                                       layernorm_eps,
                                                       max_batch_size,
                                                       max_contex_token_num,
                                                       session_len,
                                                       step_length,
                                                       start_id_,
                                                       end_id_,
                                                       cache_max_entry_count,
                                                       cache_chunk_size,
                                                       use_context_fmha,
                                                       star_coder_weights_,
                                                       tensor_para,
                                                       stream,
                                                       &cublas_wrapper,
                                                       &allocator,
                                                       is_free_buffer_after_forward,
                                                       &prop_);
        std::vector<uint32_t> output_seq_len(request_batch_size, total_output_len);

        std::unordered_map<std::string, ft::Tensor> input_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"input_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, max_input_length},
                        get_ptr<int>(input_ids)}},
            {"input_lengths",
             ft::Tensor{
                 ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size}, get_ptr<int>(input_lengths)}},
            {"output_seq_len",
             ft::Tensor{
                 ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{request_batch_size}, output_seq_len.data()}}};

        std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"output_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                        get_ptr<int>(output_ids)}},
            {"sequence_length",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width},
                        get_ptr<int>(sequence_lengths)}}};

        try {
            star_coder.forward(&output_tensors, &input_tensors);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what();
            exit(-1);
        }
        catch (...) {
            std::cout << "Runtime error";
            exit(-1);
        }
    }

private:
    const           size_t                                      head_num_;
    const           size_t                                 size_per_head_;
    const           size_t                                    inter_size_;
    const           size_t                                     layer_num_;
    const           size_t                                    vocab_size_;
    const           int                                         start_id_;
    const           int                                           end_id_;
    const           bool                                          sparse_;

    const           ft::starCoderVariantParams star_coder_variant_params_;

    std::vector<th::Tensor>                                      weights_;
    cublasLtHandle_t                                      cublasltHandle_;
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t                                  cusparseLtHandle_;
    bool                                               is_spmm_compressed = false;
#endif
    std::mutex*                                     cublas_wrapper_mutex_;
    ft::cublasAlgoMap*                                   cublas_algo_map_;
    struct          cudaDeviceProp                                  prop_;
    ft::StarCoderWeight<T>                            star_coder_weights_;
}; // FTStarCoder

class StarCoderOp: public th::jit::CustomClassHolder {
public:
    StarCoderOp(const int64_t                              head_num,
                const int64_t                         size_per_head,
                const int64_t                            inter_size,
                const int64_t                             layer_num,
                const int64_t                            vocab_size,
                const int64_t                              start_id,
                const int64_t                                end_id,
                const bool                                   sparse,
                const double                          layernorm_eps,
                const std::string                    layernorm_type,
                const std::string                   activation_type,
                const bool                  has_positional_encoding,
                const bool                has_pre_decoder_layernorm,
                const bool               has_post_decoder_layernorm,
                const bool                             has_adapters,
                const int64_t                    adapter_inter_size,
                const bool                use_attention_linear_bias,
                const vector<th::Tensor>                    weights);

    ~StarCoderOp();

    vector<th::Tensor> forward(th::Tensor                                    input_ids,
                               th::Tensor                                input_lengths,
                               const int64_t                                output_len,
                               th::optional<int64_t>                    beam_width_opt,
                               th::optional<th::Tensor>                      top_k_opt,
                               th::optional<th::Tensor>                      top_p_opt,
                               th::optional<th::Tensor> beam_search_diversity_rate_opt,
                               th::optional<th::Tensor>                temperature_opt,
                               th::optional<th::Tensor>                len_penalty_opt,
                               th::optional<th::Tensor>         repetition_penalty_opt,
                               th::optional<th::Tensor>                random_seed_opt,
                               th::optional<int64_t>          return_cum_log_probs_opt);  

private:
    const at::ScalarType             st_;
    IFStarCoder*            ftstar_coder;
    std::vector<th::Tensor>      weights;
}; // StarCoderOp

}  // namespace torch_ext