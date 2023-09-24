# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.8)

add_subdirectory(fused_multi_head_attention)

find_package(CUDAToolkit REQUIRED)

add_library(StarCoder STATIC
        StarCoder.cc
        # StarCoderContextDecoder.cc
        # StarCoderContextAttentionLayer.cc
        # StarCoderDecoderSelfAttentionLayer.cc
        # StarCoderDecoder.cc
        StarCoderWeight.cc
        StarCoderDecoderLayerWeight.cc
        # StarCoderFfnLayer.cc
        # star_coder_kernels.cu
        # star_coder_decoder_kernels.cu
        # star_coder_utils.cu)
    )
set_property(TARGET StarCoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET StarCoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(StarCoder PUBLIC CUDA::cudart
        gemm_s4_f16
        cublasMMWrapper
        DynamicDecodeLayer
        activation_kernels
        decoder_masked_multihead_attention
        bert_preprocess_kernels
        decoding_kernels
        unfused_attention_kernels
        custom_ar_kernels
        custom_ar_comm
        gpt_kernels
        tensor
        memory_utils
        nccl_utils
        cuda_utils
        logger)
        # star_coder_fmha)

if (NOT MSVC)
        add_subdirectory(flash_attention2)
        target_link_libraries(StarCoder PUBLIC flash_attention2)
endif()

add_executable(star_coder_gemm star_coder_gemm.cc)
target_link_libraries(star_coder_gemm PUBLIC CUDA::cudart gpt_gemm_func memory_utils cuda_utils logger)
install(TARGETS star_coder_gemm DESTINATION /data/yocto/slowVersion/AIXMegatronServer/FasterTransformer/build/bin)
