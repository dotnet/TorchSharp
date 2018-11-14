[![Build Status](https://migueldeicaza.visualstudio.com/TorchSharp/_apis/build/status/TorchSharp-CI)](https://migueldeicaza.visualstudio.com/TorchSharp/_build/latest?definitionId=5)

TorchSharp
==========

TorchSharp is a .NET library that provides access to the library that powers
PyTorch.  It is a work in progress, but already provides a .NET API that can
be used to perform various operations on Tensors.

Our current focus is to bind the entire C API surfaced by libtorch.

Things that you can try:

```csharp
using TorchSharp;

var x = new FloatTensor (100);   // 1D-tensor with 100 elements
FloatTensor result = new FloatTensor (100);

FloatTensor.Add (x, 23, result);

Console.WriteLine (x [12]);
```

Discussions
===========

We have a chat room here:

https://gitter.im/xamarin/TorchSharp

Status
======

Currently I am binding the Tensor APIs, opportunities to bind other APIs
are wide open.

[API Documentation](https://xamarin.github.io/TorchSharp/api/TorchSharp.html)

Running this
============
To run this, you will need libtorch and its dependencies (on Mac that
includes libc10) installed in a location that your dynamic linker can
get to.


Running On Linux (on Windows)
-----------------------------

The following was tested on Ubuntu 18.04 on the Windows Subsytem for Linux.

  1. Install the [.NET Core SDK](https://www.microsoft.com/net/download).
  2. Install [libtorch](https://pytorch.org/), and make it available to the
     dynamic linker.
  3. Run `dotnet run` in the `Tester` subfolder.
