set(SOURCE_FILES
    "common.h"
    "cpu.h"
    "cpu.cpp"
    "matrix.h"
    "cost_function.h"
    "gpu.h"
    "gpu.cu"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_30
    )
