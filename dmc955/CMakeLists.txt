cmake_minimum_required(VERSION 3.14)

find_package(Threads REQUIRED)

set(TARGET llama-adaptive-vlm)
add_executable(${TARGET} adaptive_sintel_vlm.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-adaptive-vlm)
target_link_libraries(${TARGET} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)

set(TARGET_BASE llama-baseline-vlm)
add_executable(${TARGET_BASE} baseline_vlm.cpp)
set_target_properties(${TARGET_BASE} PROPERTIES OUTPUT_NAME llama-baseline-vlm)
target_link_libraries(${TARGET_BASE} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET_BASE} PRIVATE cxx_std_17)
target_include_directories(${TARGET_BASE} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)

set(TARGET2 llama-adaptive-spatial-vlm)
add_executable(${TARGET2} adaptive_spatial_crop_vlm.cpp)
set_target_properties(${TARGET2} PROPERTIES OUTPUT_NAME llama-adaptive-spatial-vlm)
target_link_libraries(${TARGET2} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET2} PRIVATE cxx_std_17)
target_include_directories(${TARGET2} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)

set(TARGET3 llama-adaptive-patch-cache-vlm)
add_executable(${TARGET3} adaptive_patch_cache_vlm.cpp)
set_target_properties(${TARGET3} PROPERTIES OUTPUT_NAME llama-adaptive-patch-cache-vlm)
target_link_libraries(${TARGET3} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET3} PRIVATE cxx_std_17)
target_include_directories(${TARGET3} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)

set(TARGET4 llama-streaming-vlm)
add_executable(${TARGET4} streaming_vlm.cpp)
set_target_properties(${TARGET4} PROPERTIES OUTPUT_NAME llama-streaming-vlm)
target_link_libraries(${TARGET4} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET4} PRIVATE cxx_std_17)
target_include_directories(${TARGET4} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)

# Pyramid Generation (CPU-only scoring path)
set(TARGET6 llama-pyramid-gen)
add_executable(${TARGET6} generation-only/pyramid_generation.cpp)
set_target_properties(${TARGET6} PROPERTIES OUTPUT_NAME llama-pyramid-gen)
target_link_libraries(${TARGET6} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET6} PRIVATE cxx_std_17)
target_include_directories(${TARGET6} PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/generation-only)

# Scoring Strategy Sweep
set(TARGET10 llama-scoring-sweep)
add_executable(${TARGET10} generation-only/scoring_sweep.cpp)
set_target_properties(${TARGET10} PROPERTIES OUTPUT_NAME llama-scoring-sweep)
target_link_libraries(${TARGET10} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET10} PRIVATE cxx_std_17)
target_include_directories(${TARGET10} PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/generation-only)

# Keep-Ratio Sweep
set(TARGET9 llama-keep-sweep)
add_executable(${TARGET9} generation-only/keep_ratio_sweep.cpp)
set_target_properties(${TARGET9} PROPERTIES OUTPUT_NAME llama-keep-sweep)
target_link_libraries(${TARGET9} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET9} PRIVATE cxx_std_17)
target_include_directories(${TARGET9} PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/generation-only)

# Eviction Strategy Validator
set(TARGET8 llama-eviction-validator)
add_executable(${TARGET8} generation-only/eviction_validator.cpp)
set_target_properties(${TARGET8} PROPERTIES OUTPUT_NAME llama-eviction-validator)
target_link_libraries(${TARGET8} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET8} PRIVATE cxx_std_17)
target_include_directories(${TARGET8} PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/generation-only)

if(GGML_CUDA)
    enable_language(CUDA)
    set(TARGET5 llama-streaming-vlm-cuda)
    add_executable(${TARGET5} streaming_vlm.cpp kv_cache_rerotate.cu)
    set_target_properties(${TARGET5} PROPERTIES
        OUTPUT_NAME llama-streaming-vlm-cuda
        CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${TARGET5} PRIVATE common mtmd Threads::Threads)
    target_compile_features(${TARGET5} PRIVATE cxx_std_17)
    target_include_directories(${TARGET5} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
    target_compile_definitions(${TARGET5} PRIVATE HAVE_CUDA_KV_REUSE)

    # Pyramid Generation with CUDA kernels (GPU-accelerated scoring)
    set(TARGET7 llama-pyramid-gen-cuda)
    add_executable(${TARGET7}
        generation-only/pyramid_generation.cpp
        generation-only/visual_pyramid.cu)
    set_target_properties(${TARGET7} PROPERTIES
        OUTPUT_NAME llama-pyramid-gen-cuda
        CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${TARGET7} PRIVATE common mtmd Threads::Threads)
    target_compile_features(${TARGET7} PRIVATE cxx_std_17)
    target_include_directories(${TARGET7} PRIVATE
        ${CMAKE_SOURCE_DIR}/vendor
        ${CMAKE_CURRENT_SOURCE_DIR}/generation-only)
    target_compile_definitions(${TARGET7} PRIVATE HAVE_CUDA_PYRAMID)
endif()

# Window-aligned embedding cache (full / legacy / window policies, ViT shape timing)
set(TARGET11 llama-window-cache-vlm)
add_executable(${TARGET11} window_cache_vlm.cpp)
set_target_properties(${TARGET11} PROPERTIES OUTPUT_NAME llama-window-cache-vlm)
target_link_libraries(${TARGET11} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET11} PRIVATE cxx_std_17)
target_include_directories(${TARGET11} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)

# Qwen2.5-VL reference video input: frame pairs, one vision block, video positions
set(TARGET12 llama-temporal-pair-vlm)
add_executable(${TARGET12} temporal_pair_vlm.cpp)
set_target_properties(${TARGET12} PROPERTIES OUTPUT_NAME llama-temporal-pair-vlm)
target_link_libraries(${TARGET12} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET12} PRIVATE cxx_std_17)
target_include_directories(${TARGET12} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)

# Reuse potential on top of paired-frame input (stale / dropped tokens, LLM-output error)
set(TARGET13 llama-pair-reuse-vlm)
add_executable(${TARGET13} pair_reuse_vlm.cpp)
set_target_properties(${TARGET13} PROPERTIES OUTPUT_NAME llama-pair-reuse-vlm)
target_link_libraries(${TARGET13} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET13} PRIVATE cxx_std_17)
target_include_directories(${TARGET13} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
