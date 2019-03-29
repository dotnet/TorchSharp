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

Running this
============
To run this, you will need libTorchSharp, libTorch and its dependencies (on Mac that
includes libc10) installed in a location that your dynamic linker can
get to.


Running On Windows
-----------------------------

The following was tested on Windows.

  1. Install [libTorch](https://pytorch.org/), and make it available to the
     dynamic linker.
  2. Download and compile [libTorchSharp](https://github.com/interesaaat/LibTorchSharp/), and make it available to the
     dynamic linker.
  3. Run `dotnet run` in the `Test` subfolder.

Examples
===========
Porting of the more famous network architectures to TorchSharp is in progress. For the moment we only support [MNIST](https://github.com/interesaaat/TorchSharp/blob/master/Examples/MNIST.cs).
