# Jetson AGX Orin implementation (the RTX 4090 study is in dmc955/).
find_package(Threads REQUIRED)

# The pipeline: adaptive sampling, live OFA flow, window cache and the engine switches.
set(TARGET llama-jetson-vlm)
add_executable(${TARGET} jetson_vlm.cpp)
target_link_libraries(${TARGET} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor ${CMAKE_CURRENT_SOURCE_DIR})

# --live-flow: optical flow on the OFA through NVIDIA VPI (JetPack), if installed
find_package(vpi 3 QUIET PATHS /opt/nvidia/vpi3/lib/aarch64-linux-gnu/cmake)
if (vpi_FOUND)
    message(STATUS "llama-jetson-vlm: NVIDIA VPI ${vpi_VERSION} found, --live-flow enabled")
    target_link_libraries(${TARGET} PRIVATE vpi)
    target_compile_definitions(${TARGET} PRIVATE HAVE_VPI)
endif()

if (GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    # NVTX stage ranges for nsys / ncu (header-only)
    find_path(JETSON_NVTX3_DIR nvtx3/nvToolsExt.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
    if (JETSON_NVTX3_DIR)
        target_include_directories(${TARGET} PRIVATE ${JETSON_NVTX3_DIR})
        target_compile_definitions(${TARGET} PRIVATE JETSON_NVTX)
    endif()

    # Per-shape cuBLASLt algorithm tuner for GGML_CUDA_JETSON_GEMM_TABLE
    add_executable(jetson-gemm-tune tools/gemm_tune.cpp)
    target_link_libraries(jetson-gemm-tune PRIVATE CUDA::cudart CUDA::cublasLt)
    target_compile_features(jetson-gemm-tune PRIVATE cxx_std_17)
    if (JETSON_NVTX3_DIR)
        target_include_directories(jetson-gemm-tune PRIVATE ${JETSON_NVTX3_DIR})
        target_compile_definitions(jetson-gemm-tune PRIVATE JETSON_NVTX)
    endif()
endif()

# ViT encode timing / embedding dump per window-rectangle shape (vision tower only; needs NVTX)
if (JETSON_NVTX3_DIR)
    add_executable(jetson-vit-probe tools/vit_shape_probe.cpp)
    target_link_libraries(jetson-vit-probe PRIVATE mtmd llama Threads::Threads)
    target_compile_features(jetson-vit-probe PRIVATE cxx_std_17)
    target_include_directories(jetson-vit-probe PRIVATE ${JETSON_NVTX3_DIR})

    # LLM prefill of N image-sized tokens, last ubatch in its own NVTX range
    add_executable(jetson-prefill-probe tools/prefill_probe.cpp)
    target_link_libraries(jetson-prefill-probe PRIVATE llama Threads::Threads)
    target_compile_features(jetson-prefill-probe PRIVATE cxx_std_17)
    target_include_directories(jetson-prefill-probe PRIVATE ${JETSON_NVTX3_DIR})

    # Dense ViT reference: every frame of a clip encoded whole (accuracy study)
    add_executable(jetson-embed-frames tools/embed_frames.cpp)
    target_link_libraries(jetson-embed-frames PRIVATE mtmd llama Threads::Threads)
    target_compile_features(jetson-embed-frames PRIVATE cxx_std_17)

    # ViT encode and LLM prefill pipelined on two threads vs one after the other (optional stream priorities)
    add_executable(jetson-overlap-probe tools/overlap_probe.cpp)
    target_link_libraries(jetson-overlap-probe PRIVATE mtmd llama ggml Threads::Threads)
    target_compile_features(jetson-overlap-probe PRIVATE cxx_std_17)
endif()
