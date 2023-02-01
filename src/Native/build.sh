#!/usr/bin/env bash
set -e

usage()
{
    echo "Usage: $0 --arch <Architecture> "
    echo ""
    echo "Options:"
    echo "  --arch <Architecture>             Target Architecture (x64, x86)"
    echo "  --configuration <Configuration>   Build Configuration (Debug, Release)"
    echo "  --stripsymbols                    Enable symbol stripping (to external file)"
    echo "  --libtorchpath <PathToLibtorch>   Path to libtorch TorchConfig.cmake"
    exit 1
}

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
RootRepo="$DIR/../.."

__build_arch=
__strip_argument=
__libtorchpath=
__configuration=Debug
__rootBinPath="$RootRepo/bin"
__baseIntermediateOutputPath="$__rootBinPath/obj"
__versionSourceFile="$__baseIntermediateOutputPath/version.c"

while [ "$1" != "" ]; do
        lowerI="$(echo $1 | awk '{print tolower($0)}')"
        case $lowerI in
        -h|--help)
            usage
            exit 1
            ;;
        --arch)
            shift
            __build_arch=$1
            ;;
        --configuration)
            shift
            __configuration=$1
            ;;          
        --stripsymbols)
            __strip_argument="-DSTRIP_SYMBOLS=true"
            ;;
        --libtorchpath)
            shift
            __libtorchpath=$1
            ;;
        *)
        echo "Unknown argument to build.sh $1"; usage; exit 1
    esac
    shift
done

# Force the build to be release since libtorch is in release.
__cmake_defines="-DCMAKE_BUILD_TYPE=${__configuration} ${__strip_argument} -DLIBTORCH_PATH=${__libtorchpath} -DLIBTORCH_ARCH=${__build_arch}"

__IntermediatesDir="$__baseIntermediateOutputPath/$__build_arch.$__configuration/Native"
__BinDir="$__rootBinPath/$__build_arch.$__configuration/Native"

mkdir -p "$__BinDir"
mkdir -p "$__IntermediatesDir"

# Set up the environment to be used for building with clang.
if command -v "clang-6.0" > /dev/null 2>&1; then
    export CC="$(command -v clang-6.0)"
    export CXX="$(command -v clang++-6.0)"
elif command -v "clang-5.0" > /dev/null 2>&1; then
    export CC="$(command -v clang-5.0)"
    export CXX="$(command -v clang++-5.0)"
elif command -v "clang-4.0" > /dev/null 2>&1; then
    export CC="$(command -v clang-4.0)"
    export CXX="$(command -v clang++-4.0)"
elif command -v clang > /dev/null 2>&1; then
    export CC="$(command -v clang)"
    export CXX="$(command -v clang++)"
else
    echo "Unable to find Clang Compiler"
    echo "Install clang-6.0, clang-5.0, or clang-4.0"
    exit 1
fi

# Specify path to be set for CMAKE_INSTALL_PREFIX.
# This is where all built native libraries will copied to.
export __CMakeBinDir="$__BinDir"

if [ ! -f $__versionSourceFile ]; then
    __versionSourceLine="static char sccsid[] __attribute__((used)) = \"@(#)No version information produced\";"
    echo $__versionSourceLine > $__versionSourceFile
fi

OSName=$(uname -s)
case $OSName in
    Darwin)
        
        # PyTorch is specifyin options that require OpenMP support but AppleClang's  OpenMP support is lacking e.g. -fopenmp not supported
        # See    https://github.com/oneapi-src/oneDNN/issues/591 for this potential workaround, though it may be better
        # to switch to brew clang.
        #LIBOMP=/usr/local/opt/libomp
        #__cmake_defines=${__cmake_defines} -DCMAKE_CXX_FLAGS="-I$LIBOMP/include" -DCMAKE_C_FLAGS="-I$LIBOMP/include"  -DCMAKE_SHARED_LINKER_FLAGS="$LIBOMP/lib/libomp.dylib" -DCMAKE_EXE_LINKER_FLAGS="$LIBOMP/lib/libomp.dylib"
        ;;
    *)
    echo "Unsupported OS '$OSName' detected. Downloading linux-$__PKG_ARCH tools."
        OS=Linux
        __PKG_RID=linux
        ;;
esac

cd "$__IntermediatesDir"

echo "Building Machine Learning native components from $DIR to $(pwd)"
set -x # turn on trace
cmake "$DIR" -G "Unix Makefiles" $__cmake_defines
set +x # turn off trace
make install