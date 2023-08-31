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

#include "src/fastertransformer/models/multi_gpu_star_coder/ParallelStarCoderDecoderLayerWeight.h"
#include "src/fastertransformer/kernels/transpose_int8_kernels.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
ParallelStarCoderDecoderLayerWeight<T>::ParallelStarCoderDecoderLayerWeight(const int        hidden_units,
                                                                const int        inter_size,
                                                                const int        tensor_para_size,
                                                                const int        tensor_para_rank,
                                                                const int        int8_mode,
                                                                starCoderVariantParams star_coder_variant_params):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    int8_mode_(int8_mode),
    star_coder_variant_params_(star_coder_variant_params)
{
    mallocWeights();
    setWeightPtr();

    FT_CHECK_WITH_INFO(!(std::is_same<T, float>::value && int8_mode_ == 1),
                       "Weight only quant does not work with FP32 compute.");
}

template<typename T>
ParallelStarCoderDecoderLayerWeight<T>::ParallelStarCoderDecoderLayerWeight(const int int8_mode): int8_mode_(int8_mode)
{
}

template<typename T>
ParallelStarCoderDecoderLayerWeight<T>::~ParallelStarCoderDecoderLayerWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < weights_ptr.size(); i++) {
            if (weights_ptr[i] != nullptr) {
                deviceFree(weights_ptr[i]);
            }
        }

        pre_attn_layernorm_weights.beta                            = nullptr;
        pre_attn_layernorm_weights.gamma                           = nullptr;

        attention_weights.query_weight.kernel                      = nullptr;
        attention_weights.query_weight.bias                        = nullptr;
        attention_weights.key_value_weight.kernel                  = nullptr;
        attention_weights.key_value_weight.bias                    = nullptr;
        attention_weights.dense_weight.kernel                      = nullptr;
        attention_weights.dense_weight.bias                        = nullptr;

        post_attn_layernorm_weights.beta                           = nullptr;
        post_attn_layernorm_weights.gamma                          = nullptr;

        mlp_dense_weights.dense_4h_to_h_weight.kernel              = nullptr;
        mlp_dense_weights.dense_4h_to_h_weight.bias                = nullptr;
        mlp_dense_weights.dense_h_to_4h_weight.kernel              = nullptr;
        mlp_dense_weights.dense_h_to_4h_weight.bias                = nullptr;

        is_maintain_buffer = false;
    }
}

template<typename T>
void ParallelStarCoderDecoderLayerWeight<T>::copyFrom(const ParallelStarCoderDecoderLayerWeight& other)
{
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

    if (star_coder_variant_params_.has_adapters) {
        // Copy adapter biases regardless of int8 mode
        cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], star_coder_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[15], other.weights_ptr[15], hidden_units_);
        cudaD2Dcpy(weights_ptr[17], other.weights_ptr[17], star_coder_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[19], other.weights_ptr[19], hidden_units_);
    }

    if (int8_mode_ == 0) {
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);

        if (star_coder_variant_params_.has_adapters) {
            cudaD2Dcpy(weights_ptr[12],
                       other.weights_ptr[12],
                       hidden_units_ * star_coder_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(weights_ptr[14],
                       other.weights_ptr[14],
                       star_coder_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            cudaD2Dcpy(weights_ptr[16],
                       other.weights_ptr[16],
                       hidden_units_ * star_coder_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(weights_ptr[18],
                       other.weights_ptr[18],
                       star_coder_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        }
    }
    else {
        cudaD2Dcpy(
            int8_weights_ptr[0], other.int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[1], other.int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(int8_weights_ptr[2], other.int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[3], other.int8_weights_ptr[3], inter_size_ / tensor_para_size_ * hidden_units_);

        if (star_coder_variant_params_.has_adapters) {
            // Copy weights for FFN adapters after attn and regular FFN
            cudaD2Dcpy(int8_weights_ptr[4],
                       other.int8_weights_ptr[4],
                       hidden_units_ * star_coder_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[5],
                       other.int8_weights_ptr[5],
                       star_coder_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            cudaD2Dcpy(int8_weights_ptr[6],
                       other.int8_weights_ptr[6],
                       hidden_units_ * star_coder_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[7],
                       other.int8_weights_ptr[7],
                       star_coder_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        }

        if (int8_mode_ == 1) {
            cudaD2Dcpy(weight_only_scale_ptr[0], other.weight_only_scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
            cudaD2Dcpy(weight_only_scale_ptr[1], other.weight_only_scale_ptr[1], hidden_units_);
            cudaD2Dcpy(weight_only_scale_ptr[2], other.weight_only_scale_ptr[2], inter_size_ / tensor_para_size_);
            cudaD2Dcpy(weight_only_scale_ptr[3], other.weight_only_scale_ptr[3], hidden_units_);

            if (star_coder_variant_params_.has_adapters) {
                cudaD2Dcpy(weight_only_scale_ptr[4],
                           other.weight_only_scale_ptr[4],
                           star_coder_variant_params_.adapter_inter_size / tensor_para_size_);
                cudaD2Dcpy(weight_only_scale_ptr[5], other.weight_only_scale_ptr[5], hidden_units_);
                cudaD2Dcpy(weight_only_scale_ptr[6],
                           other.weight_only_scale_ptr[6],
                           star_coder_variant_params_.adapter_inter_size / tensor_para_size_);
                cudaD2Dcpy(weight_only_scale_ptr[7], other.weight_only_scale_ptr[7], hidden_units_);
            }
        }
        else if (int8_mode_ == 2) {
            cudaD2Dcpy(scale_ptr[0], other.scale_out_ptr[0], 1);
            cudaD2Dcpy(scale_inter_ptr[0], other.scale_inter_ptr[0], 3 * hidden_units_ / tensor_para_size_);
            cudaD2Dcpy(scale_out_ptr[0], other.scale_out_ptr[0], 3);

            for (int i = 1; i < 4; i++) {
                cudaD2Dcpy(scale_ptr[i], other.scale_ptr[i], 1);
                cudaD2Dcpy(scale_inter_ptr[i], other.scale_inter_ptr[i], 1);
                cudaD2Dcpy(scale_out_ptr[i], other.scale_out_ptr[i], 1);
            }
        }
    }
}

template<typename T>
ParallelStarCoderDecoderLayerWeight<T>::ParallelStarCoderDecoderLayerWeight(const ParallelStarCoderDecoderLayerWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    int8_mode_(other.int8_mode_),
    star_coder_variant_params_(other.star_coder_variant_params_)
{
    mallocWeights();
    copyFrom(other);
    setWeightPtr();
}

template<typename T>
ParallelStarCoderDecoderLayerWeight<T>&
ParallelStarCoderDecoderLayerWeight<T>::operator=(const ParallelStarCoderDecoderLayerWeight& other)
{
    hidden_units_       = other.hidden_units_;
    inter_size_         = other.inter_size_;
    tensor_para_size_   = other.tensor_para_size_;
    tensor_para_rank_   = other.tensor_para_rank_;
    int8_mode_          = other.int8_mode_;
    star_coder_variant_params_ = other.star_coder_variant_params_;

    mallocWeights();
    copyFrom(other);
    setWeightPtr();
    return *this;
}

template<typename T>
void ParallelStarCoderDecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(weights_ptr[0], {hidden_units_}, dir_path + ".input_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[1], {hidden_units_}, dir_path + ".input_layernorm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[3],
                         {3, hidden_units_ / tensor_para_size_},
                         dir_path + ".attention.query_key_value.bias." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[5], {hidden_units_}, dir_path + ".attention.dense.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[6], {hidden_units_}, dir_path + ".post_attention_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[7], {hidden_units_}, dir_path + ".post_attention_layernorm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[9],
                         {inter_size_ / tensor_para_size_},
                         dir_path + ".mlp.dense_h_to_4h.bias." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[11], {hidden_units_}, dir_path + ".mlp.dense_4h_to_h.bias.bin", model_file_type);

    if (star_coder_variant_params_.has_adapters) {
        loadWeightFromBin<T>(weights_ptr[13],
                             {star_coder_variant_params_.adapter_inter_size / tensor_para_size_},
                             dir_path + ".after_attention_adapter.dense_h_to_4h.bias."
                                 + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);

        loadWeightFromBin<T>(weights_ptr[15],
                             {hidden_units_},
                             dir_path + ".after_attention_adapter.dense_4h_to_h.bias.bin",
                             model_file_type);

        loadWeightFromBin<T>(weights_ptr[17],
                             {star_coder_variant_params_.adapter_inter_size / tensor_para_size_},
                             dir_path + ".after_ffn_adapter.dense_h_to_4h.bias." + std::to_string(tensor_para_rank_)
                                 + ".bin",
                             model_file_type);

        loadWeightFromBin<T>(
            weights_ptr[19], {hidden_units_}, dir_path + ".after_ffn_adapter.dense_4h_to_h.bias.bin", model_file_type);
    }

    // Load weights for StarCoder
    if (int8_mode_ == 0) {
        loadWeightFromBin<T>(weights_ptr[2],
                             {hidden_units_, 3 * hidden_units_ / tensor_para_size_},
                             dir_path + ".attention.query_key_value.weight." + std::to_string(tensor_para_rank_)
                                 + ".bin",
                             model_file_type);

        loadWeightFromBin<T>(weights_ptr[4],
                             {hidden_units_ / tensor_para_size_, hidden_units_},
                             dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);

        loadWeightFromBin<T>(weights_ptr[8],
                             {hidden_units_, inter_size_ / tensor_para_size_},
                             dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);

        loadWeightFromBin<T>(weights_ptr[10],
                             {inter_size_ / tensor_para_size_, hidden_units_},
                             dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);

        // Load adapter weights if required.
        if (star_coder_variant_params_.has_adapters) {
            loadWeightFromBin<T>(weights_ptr[12],
                                 {hidden_units_, star_coder_variant_params_.adapter_inter_size / tensor_para_size_},
                                 dir_path + ".after_attention_adapter.dense_h_to_4h.weight."
                                     + std::to_string(tensor_para_rank_) + ".bin",
                                 model_file_type);

            loadWeightFromBin<T>(weights_ptr[14],
                                 {star_coder_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
                                 dir_path + ".after_attention_adapter.dense_4h_to_h.weight."
                                     + std::to_string(tensor_para_rank_) + ".bin",
                                 model_file_type);

            loadWeightFromBin<T>(weights_ptr[16],
                                 {hidden_units_, star_coder_variant_params_.adapter_inter_size / tensor_para_size_},
                                 dir_path + ".after_ffn_adapter.dense_h_to_4h.weight."
                                     + std::to_string(tensor_para_rank_) + ".bin",
                                 model_file_type);

            loadWeightFromBin<T>(weights_ptr[18],
                                 {star_coder_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
                                 dir_path + ".after_ffn_adapter.dense_4h_to_h.weight."
                                     + std::to_string(tensor_para_rank_) + ".bin",
                                 model_file_type);
        }
    }
    else if (int8_mode_ == 1) {
        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[0],
                                                     weight_only_scale_ptr[0],
                                                     {hidden_units_, 3 * hidden_units_ / tensor_para_size_},
                                                     dir_path + ".attention.query_key_value.weight."
                                                         + std::to_string(tensor_para_rank_) + ".bin",
                                                     model_file_type);

        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[1],
                                                     weight_only_scale_ptr[1],
                                                     {hidden_units_ / tensor_para_size_, hidden_units_},
                                                     dir_path + ".attention.dense.weight."
                                                         + std::to_string(tensor_para_rank_) + ".bin",
                                                     model_file_type);

        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[2],
                                                     weight_only_scale_ptr[2],
                                                     {hidden_units_, inter_size_ / tensor_para_size_},
                                                     dir_path + ".mlp.dense_h_to_4h.weight."
                                                         + std::to_string(tensor_para_rank_) + ".bin",
                                                     model_file_type);

        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[3],
                                                     weight_only_scale_ptr[3],
                                                     {inter_size_ / tensor_para_size_, hidden_units_},
                                                     dir_path + ".mlp.dense_4h_to_h.weight."
                                                         + std::to_string(tensor_para_rank_) + ".bin",
                                                     model_file_type);

        // Load adapter weights if required.
        if (star_coder_variant_params_.has_adapters) {
            loadWeightFromBinAndQuantizeForWeightOnly<T>(
                int8_weights_ptr[4],
                weight_only_scale_ptr[4],
                {hidden_units_, star_coder_variant_params_.adapter_inter_size / tensor_para_size_},
                dir_path + ".after_attention_adapter.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_)
                    + ".bin",
                model_file_type);

            loadWeightFromBinAndQuantizeForWeightOnly<T>(
                int8_weights_ptr[5],
                weight_only_scale_ptr[5],
                {star_coder_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
                dir_path + ".after_attention_adapter.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_)
                    + ".bin",
                model_file_type);

            loadWeightFromBinAndQuantizeForWeightOnly<T>(
                int8_weights_ptr[6],
                weight_only_scale_ptr[6],
                {hidden_units_, star_coder_variant_params_.adapter_inter_size / tensor_para_size_},
                dir_path + ".after_ffn_adapter.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                model_file_type);

            loadWeightFromBinAndQuantizeForWeightOnly<T>(
                int8_weights_ptr[7],
                weight_only_scale_ptr[7],
                {star_coder_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
                dir_path + ".after_ffn_adapter.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                model_file_type);
        }
    }
    else if (int8_mode_ == 2) {
        const auto                     tp_rank = std::to_string(tensor_para_rank_);
        const std::vector<std::string> weight_list{
            "attention.query_key_value", "attention.dense", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"};
        const std::vector<std::vector<size_t>> shape_list{{hidden_units_, 3 * hidden_units_ / tensor_para_size_},
                                                          {hidden_units_ / tensor_para_size_, hidden_units_},
                                                          {hidden_units_, inter_size_ / tensor_para_size_},
                                                          {inter_size_ / tensor_para_size_, hidden_units_}};
        for (int i = 0; i < weight_list.size(); i++) {
            loadWeightFromBin<int8_t>(int8_weights_ptr[i],
                                      shape_list[i],
                                      dir_path + "." + weight_list[i] + ".weight.int8." + tp_rank + ".bin",
                                      FtCudaDataType::INT8);

            const std::pair<std::vector<std::vector<float*>*>, std::vector<std::string>> arg_pair{
                {&scale_ptr, &scale_inter_ptr, &scale_out_ptr}, {"scale", "scale_inter", "scale_out"}};
            for (int j = 0; j < arg_pair.first.size(); j++) {
                size_t num_elems = 1;
                // attention.qkv scale_inter has 3 weights for Q, K and V
                // attention.qkv scale_out has 3 weights for Q, K and V, duplicated along hidden_units dim
                if (i == 0 && j == 1) {
                    num_elems = 3 * hidden_units_ / tensor_para_size_;
                }
                else if (i == 0 && j == 2) {
                    num_elems = 3;
                }

                loadWeightFromBin<float>((*arg_pair.first[j])[i],
                                         {num_elems},
                                         dir_path + "." + weight_list[i] + "." + arg_pair.second[j] + ".bin",
                                         FtCudaDataType::FP32);
            }
        }
        transposeWeight();
    }
}

template<typename T>
void ParallelStarCoderDecoderLayerWeight<T>::setWeightPtr()
{
    pre_attn_layernorm_weights.gamma                           = weights_ptr[ 0];
    pre_attn_layernorm_weights.beta                            = weights_ptr[ 1];

    attention_weights.query_weight.kernel                      = weights_ptr[ 2];
    attention_weights.key_value_weight.kernel                  = weights_ptr[ 3];
    attention_weights.query_weight.bias                        = weights_ptr[ 4];
    attention_weights.key_value_weight.bias                    = weights_ptr[ 5];
    attention_weights.dense_weight.kernel                      = weights_ptr[ 6];
    attention_weights.dense_weight.bias                        = weights_ptr[ 7];

    post_attn_layernorm_weights.gamma                          = weights_ptr[ 8];
    post_attn_layernorm_weights.beta                           = weights_ptr[ 9];

    mlp_dense_weights.dense_h_to_4h_weight.kernel              = weights_ptr[10];
    mlp_dense_weights.dense_h_to_4h_weight.bias                = weights_ptr[11];
    mlp_dense_weights.dense_4h_to_h_weight.kernel              = weights_ptr[12];
    mlp_dense_weights.dense_4h_to_h_weight.bias                = weights_ptr[13];

    is_maintain_buffer = true;
}

template<typename T>
void ParallelStarCoderDecoderLayerWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[ 0], 6144);      
    deviceMalloc(&weights_ptr[ 1], 6144);         
    deviceMalloc(&weights_ptr[ 2], 6144 * 6144); 
    deviceMalloc(&weights_ptr[ 3], 6144 * 256);  
    deviceMalloc(&weights_ptr[ 4], 6144);         
    deviceMalloc(&weights_ptr[ 5], 256);        
    deviceMalloc(&weights_ptr[ 6], 6144* 6144); 
    deviceMalloc(&weights_ptr[ 7], 6144);     
    deviceMalloc(&weights_ptr[ 8], 6144);     
    deviceMalloc(&weights_ptr[ 9], 6144);        
    deviceMalloc(&weights_ptr[10], 6144 * 24576);   
    deviceMalloc(&weights_ptr[11], 24576);         
    deviceMalloc(&weights_ptr[12], 6144 * 24576);  
    deviceMalloc(&weights_ptr[13], 6144);         
}

#ifdef SPARSITY_ENABLED
template<typename T>
void ParallelStarCoderDecoderLayerWeight<T>::compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim)
{
    hidden_units_ = hidden_dim;
    inter_size_   = 4 * hidden_units_;

    const size_t num_sparse_weights            = 8;
    size_t       shapes[num_sparse_weights][2] = {
              {hidden_units_, 3 * hidden_units_ / tensor_para_size_},
              {hidden_units_ / tensor_para_size_, hidden_units_},
              {hidden_units_, inter_size_ / tensor_para_size_},
              {inter_size_ / tensor_para_size_, hidden_units_},
              {hidden_units_, star_coder_variant_params_.adapter_inter_size / tensor_para_size_},
              {star_coder_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
              {hidden_units_, star_coder_variant_params_.adapter_inter_size / tensor_para_size_},
              {star_coder_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_}};

    const T* dense_weights[num_sparse_weights] = {self_attention_weights.query_weight.kernel,
                                                  self_attention_weights.attention_output_weight.kernel,
                                                  ffn_weights.intermediate_weight.kernel,
                                                  ffn_weights.output_weight.kernel,
                                                  after_attention_adapter_weights.intermediate_weight.kernel,
                                                  after_attention_adapter_weights.output_weight.kernel,
                                                  after_ffn_adapter_weights.intermediate_weight.kernel,
                                                  after_ffn_adapter_weights.output_weight.kernel};

    size_t real_num_sparse_weights = star_coder_variant_params_.has_adapters ? num_sparse_weights : (num_sparse_weights - 4);
    for (size_t i = 0; i < real_num_sparse_weights; ++i) {
        int    m               = shapes[i][1];
        int    k               = shapes[i][0];
        size_t compressed_size = cublas_wrapper.getSparseMatrixSize(m, k);
        deviceMalloc(&sp_weights_ptr[i], static_cast<int>(compressed_size), false);
        cublas_wrapper.compressMatrix(dense_weights[i], sp_weights_ptr[i], m, k);
    }

    self_attention_weights.query_weight.sp_kernel                 = sp_weights_ptr[0];
    self_attention_weights.attention_output_weight.sp_kernel      = sp_weights_ptr[1];
    ffn_weights.intermediate_weight.sp_kernel                     = sp_weights_ptr[2];
    ffn_weights.output_weight.sp_kernel                           = sp_weights_ptr[3];
    after_attention_adapter_weights.intermediate_weight.sp_kernel = sp_weights_ptr[4];
    after_attention_adapter_weights.output_weight.sp_kernel       = sp_weights_ptr[5];
    after_ffn_adapter_weights.intermediate_weight.sp_kernel       = sp_weights_ptr[6];
    after_ffn_adapter_weights.output_weight.sp_kernel             = sp_weights_ptr[7];
    is_maintain_sp_buffer                                         = true;
}
#endif

template<typename T>
void ParallelStarCoderDecoderLayerWeight<T>::transposeWeight()
{
    const auto                             tp = tensor_para_size_;
    const std::vector<std::vector<size_t>> shape_list{{hidden_units_, 3 * hidden_units_ / tp},
                                                      {hidden_units_ / tp, hidden_units_},
                                                      {hidden_units_, inter_size_ / tp},
                                                      {inter_size_ / tp, hidden_units_}};

    const auto max_size =
        sizeof(int8_t) * std::max(3 * hidden_units_ * hidden_units_ / tp, hidden_units_ * inter_size_ / tp);

    int8_t* transpose_temp;
    cudaMalloc(&transpose_temp, max_size);

    for (int i = 0; i < shape_list.size(); i++) {
        invokeTransposeInt8Tensor({MEMORY_GPU, TYPE_INT8, {shape_list[i][1], shape_list[i][0]}, transpose_temp},
                                  {MEMORY_GPU, TYPE_INT8, shape_list[i], int8_weights_ptr[i]},
                                  stream_);
        cudaD2Dcpy(int8_weights_ptr[i], transpose_temp, shape_list[i][0] * shape_list[i][1]);
    }

    cudaFree(transpose_temp);
}

template struct ParallelStarCoderDecoderLayerWeight<float>;
template struct ParallelStarCoderDecoderLayerWeight<half>;
#ifdef ENABLE_BF16
template struct ParallelStarCoderDecoderLayerWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
