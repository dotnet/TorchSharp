[![Build Status (master)](https://dev.azure.com/migueldeicaza/TorchSharp/_apis/build/status/xamarin.TorchSharp?branchName=master)](https://dev.azure.com/migueldeicaza/TorchSharp/_build/latest?definitionId=17&branchName=master)  

TorchSharp
==========

TorchSharp is a .NET library that provides access to the library that powers
PyTorch.  It is a work in progress, but already provides a .NET API that can
be used to perform (1) various operations on ATen Tensors; (2) scoring of 
TorchScript models; (3) Training of simple neural networks.

Our current focus is to bind the entire API surfaced by libtorch.

Things that you can try:

```csharp
using TorchSharp;
using TorchSharp.Tensor;
using TorchSharp.NN;
using static TorchSharp.Tensor.Modules;

var lin1 = Linear(1000, 100);
var lin2 = Linear(100, 10);
var seq = Sequential(lin1, Relu(), lin2);

var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

double learning_rate = 0.00004f;
float prevLoss = float.MaxValue;
var optimizer = Optimizer.Adam(seq.Parameters(), learning_rate);
var loss = Losses.MSE(NN.Reduction.Sum);

for (int i = 0; i < 10; i++)
{
    var eval = seq.Forward(x);
    var output = loss(eval, y);
    var lossVal = output.DataItem<float>();

    Assert.True(lossVal < prevLoss);
    prevLoss = lossVal;

    optimizer.ZeroGrad();

    output.Backward();

    optimizer.Step();
}
```

Discussions
===========

We have a chat room on Gitter [![Gitter](https://badges.gitter.im/xamarin/TorchSharp.svg)](https://gitter.im/xamarin/TorchSharp?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


Building
============


Windows
-----------------------------

Requirements:
- Visual Studio
- git
- cmake (tested with 3.14)

Commands:
- Building: `build.cmd build` (can use  `dotnet build` after first time)
- Building from Visual Studio: first build using the command line
- See all configurations: `build.cmd -?`
- Run tests from command line: `dotnet test`


Linux/Mac
-----------------------------
Requirements:
- requirements to run .NET Core 2.0
- git
- cmake (tested with 3.14)
- clang 4.x +

Example to fulfill the requirements in Ubuntu 16:
```
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main"
sudo apt-get -y update
sudo apt-get -y install clang-6.0 git cmake libunwind8 curl libssl1.0.0 libomp-dev
```

Commands:
- Building: `./build.sh`
- Building from Visual Studio: first build using the command line
- See all configurations: `./build.sh -?`
- Run tests from command line: `dotnet test`
- Build packages: `dotnet pack`



Building packages
------------------------

The managed package can be built with `dotnet pack`, e.g.

    dotnet pack /p:SkipCuda=true

Locally built packages have names like this, names update every day.  If repeatedly rebuilding them locally you may have to remove them
from your local `.nuget` package cache.

    bin/packages/Debug/libtorch-cuda-10.2.0.3.0-local-Debug-20200520.nupkg
    bin/packages/Debug/libtorch-cpu.0.3.0-local-Debug-20200520.nupkg
    bin/packages/Debug/TorchSharp.0.3.0-local-Debug-20200520.nupkg

    bin/packages/Release/libtorch-cuda-10.2.0.3.0-local-Release-20200520.nupkg
    bin/packages/Release/libtorch-cpu.0.3.0-local-Release-20200520.nupkg
    bin/packages/Release/TorchSharp.0.3.0-local-Release-20200520.nupkg


Complete libtorch-cpu packages can't be built using your local machine alone, since they won't contain the
full range of native bits. Instead they are built using Azure Pipelines.

An ephemeral feed of packages from CI is available 

* View link: https://dev.azure.com/migueldeicaza/TorchSharp/_packaging?_a=feed&feed=packages%40Local
* Nuget link: https://pkgs.dev.azure.com/migueldeicaza/TorchSharp/_packaging/packages%40Local/nuget/v3/index.json. 



Updating PyTorch version
------------------------

This project grabs LibTorch and makes a C API wrapper for it, then calls these from C#.

See https://pytorch.org/get-started/locally/ for download links.

For example Linux, LibTorch 1.5.0 uses link

    https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.5.0%2Bcpu.zip

To update the version, update these:

    <LibtorchVersion>1.5.0</LibtorchVersion>

Then run these to test downloads and update SHA hashes for the various LibTorch downloads:

    msbuild src\Redist\libtorch-cuda-10.2\libtorch-cuda-10.2.proj /p:UpdateSHA=true /p:TargetOS=linux /t:Build
    msbuild src\Redist\libtorch-cuda-10.2\libtorch-cuda-10.2.proj /p:UpdateSHA=true /p:TargetOS=windows /t:Build

    msbuild src\Redist\libtorch-cpu\libtorch-cpu.proj /p:UpdateSHA=true /p:TargetOS=linux /t:Build
    msbuild src\Redist\libtorch-cpu\libtorch-cpu.proj /p:UpdateSHA=true /p:TargetOS=windows /t:Build
    msbuild src\Redist\libtorch-cpu\libtorch-cpu.proj /p:UpdateSHA=true /p:TargetOS=mac /t:Build

You must also update the "FilesFromArchive= ..." entries under src\Redist projects. Check the contents
of the unzip of the archive, e.g.

     bin\obj\x86.Debug\libtorch-cpu\libtorch-shared-with-deps-1.5.0%2Bcpu\libtorch\lib


Updating package version for new release
-----------------------------

To change the package version update this [file](https://github.com/xamarin/TorchSharp/blob/master/build/BranchInfo.props).

Sample command: `./build.cmd pack`

GPU support
============
For GPU support it is required to install CUDA 9.0 and make it available to the dynamic linker.

Examples
===========
Porting of the more famous network architectures to TorchSharp is in progress. For the moment we only support [MNIST](https://github.com/xamarin/TorchSharp/blob/master/src/Examples/MNIST.cs) and [AlexNet](https://github.com/xamarin/TorchSharp/blob/master/src/Examples/AlexNet.cs)
