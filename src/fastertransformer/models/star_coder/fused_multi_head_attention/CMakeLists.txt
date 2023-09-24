
cmake_minimum_required(VERSION 3.8)

add_library(star_coder_fmha STATIC star_coder_flash_attention_kernel.cu)
target_include_directories(star_coder_fmha PRIVATE ${CUTLASS_DIR}/examples)
target_link_libraries(star_coder_fmha PRIVATE nvidia::cutlass::cutlass)
set_property(TARGET star_coder_fmha PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET star_coder_fmha PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
