[![Build Status](https://migueldeicaza.visualstudio.com/TorchSharp/_apis/build/status/TorchSharp-CI)](https://migueldeicaza.visualstudio.com/TorchSharp/_build/latest?definitionId=5)

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
