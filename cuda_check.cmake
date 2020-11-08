# Find CUDA
find_package(CUDA)

if (CUDA_FOUND)
  #Get CUDA compute capability
  set(OUTPUTFILE ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cuda_script) # No suffix required
  set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_cuda.cu)
  execute_process(COMMAND ${CMAKE_CUDA_COMPILER} -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
  execute_process(COMMAND ${OUTPUTFILE}
                  RESULT_VARIABLE CUDA_RETURN_CODE
                  OUTPUT_VARIABLE ARCH)
  execute_process(COMMAND rm ${OUTPUTFILE})

  if(${CUDA_RETURN_CODE} EQUAL 0)
    set(CUDA_SUCCESS "TRUE")
  else()
    set(CUDA_SUCCESS "FALSE")
  endif()

  if (${CUDA_SUCCESS})
    message(STATUS "CUDA Architecture: sm_${ARCH}")
    message(STATUS "CUDA Version: ${CUDA_VERSION_STRING}")
    message(STATUS "CUDA Path: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA Performance Primitives: ${CUDA_npp_LIBRARY}")

    set(CUDA_NVCC_ARCH_FLAGS "${ARCH}")
  else()
    message(WARNING ${ARCH})
  endif()
endif()
