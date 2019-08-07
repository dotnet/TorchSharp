[![Build Status](https://migueldeicaza.visualstudio.com/TorchSharp/_apis/build/status/TorchSharp-CI)](https://migueldeicaza.visualstudio.com/TorchSharp/_build/latest?definitionId=5)

TorchSharp
==========

TorchSharp is a .NET library that provides access to the library that powers
PyTorch.  It is a work in progress, but already provides a .NET API that can
be used to perform (1) various operations on ATen Tensors; (2) scoring of 
TorchScript models; (3) Training of simple neural networks.

Our current focus is to bind the entire API surfaced by libtorch.

Things that you can try:

```csharp
using AtenSharp;

var x = new FloatTensor (100);   // 1D-tensor with 100 elements
FloatTensor result = new FloatTensor (100);

FloatTensor.Add (x, 23, result);

Console.WriteLine (x [12]);
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
- Building: `build.cmd`
- Building from Visual Studio: first build using the command line
- See all configurations: `build.cmd -?`
- Run tests from command line: `build.cmd -runtests`
- Build packages: `build.cmd -buildpackages`


Linux/Mac
-----------------------------
Requirements:
- requirements to run .NET Core 2.0
- git
- cmake (tested with 3.14)
- clang 3.9

Example to fulfill the requirements in Ubuntu 16:
```
sudo apt-get update
sudo apt-get install git clang cmake libunwind8 curl
sudo apt-get install libssl1.0.0
sudo apt-get install libomp-dev
```

Commands:
- Building: `./build.sh`
- Building from Visual Studio: first build using the command line
- See all configurations: `./build.sh -?`
- Run tests from command line: `./build.sh -runtests`
- Build packages: `./build.sh -buildpackages`

Updating package version for new release
-----------------------------
To change the pacakage version update this [file](https://github.com/xamarin/TorchSharp/blob/master/build/BranchInfo.props).
Everything is currently considered in preview.

Use the following two MSBuild arguments in order to control the -preview and the build numbers in the name of the nuget packages produced (use one of the two generally):

|Name | Value| Example Version Output|
|---|---|---|
|StabilizePackageVersion |  true  | 1.0.0|
|IncludeBuildNumberInPackageVersion | false | 1.0.0-preview|

Sample command: `./build.cmd -release -buildpackages -- /p:StabilizePackageVersion=true`

Examples
===========
Porting of the more famous network architectures to TorchSharp is in progress. For the moment we only support [MNIST](https://github.com/xamarin/TorchSharp/blob/master/src/Examples/MNIST.cs) and [AlexNet](https://github.com/xamarin/TorchSharp/blob/master/src/Examples/AlexNet.cs)
