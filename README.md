[![Build Status](https://donsyme.visualstudio.com/TorchSharp/_apis/build/status/xamarin.TorchSharp?branchName=master)](https://donsyme.visualstudio.com/TorchSharp/_build/latest?definitionId=1&branchName=master)

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

# Discussions

We have a chat room on Gitter [![Gitter](https://badges.gitter.im/xamarin/TorchSharp.svg)](https://gitter.im/xamarin/TorchSharp?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


# Building


## Windows

Requirements:
- Visual Studio
- git
- cmake (tested with 3.14)

Commands:
- Building: `build.cmd build` (can use  `dotnet build` after first time)
- Building from Visual Studio: first build using the command line
- Run tests from command line: `dotnet test`


## Linux/Mac

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
- Run tests from command line: `dotnet test`
- Build packages: `dotnet pack`


## Packages

An ephemeral feed of packages from CI is available 

* View link: https://donsyme.visualstudio.com/TorchSharp/_packaging?_a=feed&feed=packages2
* Nuget feed: https://donsyme.pkgs.visualstudio.com/TorchSharp/_packaging/packages2/nuget/v3/index.json


## Building the TorchSharp package

The managed package can be built with `dotnet pack`, e.g.

    ./build.cmd pack

or just 

    dotnet pack

Locally built packages have names like this, names update every day.  If repeatedly rebuilding them locally you may have to remove them
from your local `.nuget` package cache.

    bin/packages/Debug/TorchSharp.0.3.0-local-Debug-20200520.nupkg
    bin/packages/Release/TorchSharp.0.3.0-local-Release-20200520.nupkg

To change the TorchSharp package version update this [file](https://github.com/xamarin/TorchSharp/blob/master/build/BranchInfo.props).

## Making releases of the TorchSharp package

The TorchSharp package is pushed to nuget.org either manually or as part of Azure DevOps CI release pipeline, see below.

# The libtorch packages

The libtorch packages are huge (~1.6GB compressed) and cause a lot of problems to make and delier due to nuget package size restrictions.
These problems include:

1. A massive 1GB binary in the linux CUDA package and multiple 0.5GB binaries in Windows CUDA package

2. Size limitations of about ~500MB on nuget packages on the Azure DevOps CI system and about ~250MB on `nuget.org`

4. Regular download/upload failures on these systems due to network interruptions for packages of this size

5. 10GB VM image size restrictions for the containers userd to build these packages in the Azure DevOps CI system, we can easily run out of room.

6. Complete libtorch-cpu packages can't be built using your local machine alone, since they won't contain the
   full range of native bits. Instead they are built using Azure Pipelines.

For this reason, we do the following

1. The head, referenceable packages that deliver a functioning runtime are any of:

   libtorch-cpu
   libtorch-cuda-10.2
   libtorch-cuda-10.2-linux-x64
   libtorch-cuda-10.2-win-x64

2. These packages are combo packages that reference multiple parts.  The parts are **not** independently useful.

3. Some parts deliver a single vast file via `primary` and `fragment` packages.  A build task is then used to "stitch" these files back together 
   to one file on the target machine with a SHA check.  This is a hack but there is no other realistic way to deliver
   these vast files as packages (the alternative is to abandon packaging and require a manual
   install/detect/link of PyTorch CUDA on all downstream systems, whcih is extremely problematic
   for many practical reasons).

4. The `libtorch-*` packages are built in Azure DevOps CI
   [using this build pipeline](https://donsyme.visualstudio.com/TorchSharp/_build?definitionId=1&_a=summary) but only in master
   branch and only when `<BuildLibTorchPackages>true</BuildLibTorchPackages>` is set in that branch.  You must currently
   manually set this, increment `LibTorchPackageVersion`, do a push to master and the packages will build.  This process could be adjusted
   but at least gets us off the ground.

5. After a successful build, the `libtorch-*` packages can be trialled using the package feed from CI (see above).  When
   they are appropriate they can be  pushed to nuget using
   [this manually invoked release pipeline](https://donsyme.visualstudio.com/TorchSharp/_release?_a=releases&view=mine&definitionId=1) in
   Azure DevOps CI (so they don't have to be manually downloaded and pushed to `nuget.org`)

   a. [Go to release pipeline](https://donsyme.visualstudio.com/TorchSharp/_release?_a=releases&view=mine&definitionId=1)

   b. Press 'New Release'

   c. Select the successful master CI build that includes the `libtorch` packages, create the release and wait for it to finish. You should
      see `Initialize job`, `Download artifact - _xamarin.TorchSharp - packages`, `NuGet push`, `Finalize Job` succeeded.

   d. All packages should now be pushed to `nuget.org` and will appear after indexing.


### Updating PyTorch version for libtorch packages

This project grabs LibTorch and makes a C API wrapper for it, then calls these from C#. When updating to a newer
version of PyTorch then quite a lot of careful work needs to be done.

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

You must also adjust the set of binaries referenced for tests, see various files under `tests`.

Examples
===========

Porting of the more famous network architectures to TorchSharp is in progress. For the moment we only support [MNIST](https://github.com/xamarin/TorchSharp/blob/master/src/Examples/MNIST.cs) and [AlexNet](https://github.com/xamarin/TorchSharp/blob/master/src/Examples/AlexNet.cs)

[DiffSharp](https://github.com/DiffSharp/DiffSharp/) also uses this repository extensively and has been a major factor in iterating support.
