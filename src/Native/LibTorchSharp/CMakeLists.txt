project(LibTorchSharp)

find_package(CUDA)
if(CUDA_FOUND)
	include_directories(${CUDA_INCLUDE_DIRS})
	link_directories(${CUDA_LIBRARY_DIRS})
	add_compile_definitions(TORCHSHARP_CUDA_TOOLKIT_FOUND)
endif()

add_compile_definitions(NOMINMAX)


#add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
# ensure that PyTorch is told to use NVTX3 headers
#target_compile_definitions(CUDA::nvToolsExt INTERFACETORCH_CUDA_USE_NVTX3)
#target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)



if(APPLE AND NOT LIBTORCH_ARCH STREQUAL "arm64")
 include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
 link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
endif()

#set(LIBTORCH_PATH "K:/FrameworksForC/LibTorch/libtorch-win-shared-with-deps-2.6.0+cu126")
find_package(Torch REQUIRED PATHS ${LIBTORCH_PATH})
#find_package(Torch CONFIG)

set(SOURCES
    cifar10.h
	crc32c.h
	THSAmp.h
    THSAutograd.h
    THSBFloat16.h
	THSCuda.h
    THSData.h
    THSJIT.h
    THSNN.h
	THSStorage.h
    THSTensor.h
    THSTorch.h
	THSVision.h
    Utils.h
    cifar10.cpp
	crc32c.c
	THSActivation.cpp
	THSAmp.cpp
    THSAutograd.cpp
	THSBFloat16.cpp
	THSCuda.cpp
	THSConvolution.cpp
    THSData.cpp
	THSFFT.cpp
    THSJIT.cpp
	THSLinearAlgebra.cpp
	THSLoss.cpp
	THSModule.cpp
    THSNN.cpp
	THSNormalization.cpp
	THSOptimizers.cpp
	THSRandom.cpp
	THSSpecial.cpp
	THSStorage.cpp
    THSTensor.cpp
	THSTensorConv.cpp
	THSTensorFactories.cpp
	THSTensorMath.cpp
    THSTorch.cpp
	THSVision.cpp
    Utils.cpp)

if(NOT WIN32)
    list(APPEND SOURCES ${VERSION_FILE_PATH})
    if(NOT APPLE)
        SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
        SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
        SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
        SET(CMAKE_INSTALL_RPATH "$ORIGIN/")

    else()
        SET(CMAKE_MACOSX_RPATH TRUE)
    endif()

endif()

# Enable cross compilation for arm64/x64 on macOS
if(APPLE)
	if(LIBTORCH_ARCH STREQUAL "arm64")
		set(CMAKE_OSX_ARCHITECTURES "arm64")
	else()
		set(CMAKE_OSX_ARCHITECTURES "x86_64")
	endif()
endif()

# Add libTorch bindings
include_directories(${TORCH_INCLUDE_DIRS})

add_library(LibTorchSharp SHARED ${SOURCES} ${RESOURCES})

if(CUDA_FOUND)
target_link_libraries(LibTorchSharp ${CUDA_LIBRARIES})
endif()

target_link_libraries(LibTorchSharp ${TORCH_LIBRARIES})

set_property(TARGET LibTorchSharp PROPERTY CXX_STANDARD 14)

if(APPLE)
    set_target_properties(LibTorchSharp PROPERTIES INSTALL_RPATH "@loader_path;@executable_path;")
endif()

# Set C++ standard
set_target_properties(LibTorchSharp PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

if(NOT WIN32)
    # Add C++ specific compile options
    target_compile_options(LibTorchSharp PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
    )
endif()


install_library_and_symbols (LibTorchSharp)
