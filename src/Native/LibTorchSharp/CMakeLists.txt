project(LibTorchSharp)

if(APPLE)
 include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
 link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
endif()
find_package(Torch REQUIRED PATHS ${LIBTORCH_PATH})

set(SOURCES
    cifar10.h
    THSAutograd.h
    THSData.h
    THSJIT.h
    THSNN.h
    THSTensor.h
    THSTorch.h
    Utils.h
    cifar10.cpp
    THSAutograd.cpp
    THSData.cpp
    THSJIT.cpp
    THSNN.cpp
    THSTensor.cpp
    THSTorch.cpp
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

# Add libTorch bindings
include_directories(${TORCH_INCLUDE_DIRS})

add_library(LibTorchSharp SHARED ${SOURCES} ${RESOURCES})

# Make the libLibTorchSharp.so independent of installed libc++ by static linking
# see
#   https://github.com/xamarin/TorchSharp/issues/218
#   https://stackoverflow.com/questions/46809303/how-to-static-linking-to-glibc-in-cmake
if(NOT WIN32)
    target_link_libraries(LibTorchSharp ${TORCH_LIBRARIES} -static-libstdc++)
endif()

set_property(TARGET LibTorchSharp PROPERTY CXX_STANDARD 14)

if(APPLE)
    set_target_properties(LibTorchSharp PROPERTIES INSTALL_RPATH "@loader_path;@executable_path;")
endif()

install_library_and_symbols (LibTorchSharp)
