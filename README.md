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

We have a chat room here:

https://gitter.im/xamarin/TorchSharp

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

Command:
- Building: `./build.sh`
- Building from Visual Studio: first build using the command line
- See all configurations: `./build.sh -?`
- Run tests from command line: `./build.sh -runtests`
- Build packages: `./build.sh -buildpackages`

Examples
===========
Porting of the more famous network architectures to TorchSharp is in progress. For the moment we only support [MNIST](https://github.com/interesaaat/TorchSharp/blob/master/Examples/MNIST.cs).
