[![Build Status](https://dotnet.visualstudio.com/TorchSharp/_apis/build/status/dotnet.TorchSharp?branchName=main)](https://donsyme.visualstudio.com/TorchSharp/_build/latest?definitionId=174&branchName=main)

__TorchSharp is now in the .NET Foundation!__

If you are using TorchSharp from NuGet, you should be using a version >= 0.93.1 of TorchSharp, and >= 1.9.0.11 of the libtorch-xxx redistributable packages.


__NOTE:__

In PR 302, significant changes were made to the TorchSharp API, aligning more closely with the Pytorch APIs. This was a massive breaking change. We apologize for any extra work this may cause, but we believe that what was done is in the best long-term interest of TorchSharp users.

In PR 354, further significant changes were made, again aligning with the Pytorch APIs. It is also a massively braking change. We removed the {IntNN|FloatNN|ComplexNN}Tensor.* APIs, which had no parallel in PyTorch. Once again, we apologize for any extra work this may cause, but we believe that what was done is in the best long-term interest of TorchSharp users.

# TorchSharp

TorchSharp is a .NET library that provides access to the library that powers PyTorch. It is part of the .NET Foundation.

The focus is to bind the API surfaced by libtorch with a particular focus on tensors. The design intent is to stay as close as possible to the Pytorch experience, while still taking advantage of the benefits of the .NET static type system where it makes sense. For example: method overloading is relied on when Pytorch defines multiple valid types for a particular parameter.

The technology is a "wrapper library": no more, no less. [DiffSharp](https://github.com/DiffSharp/DiffSharp/) uses this
repository extensively and has been a major factor in iterating support.

Things that you can try:

```csharp
using TorchSharp;
using static torch.nn;

var lin1 = Linear(1000, 100);
var lin2 = Linear(100, 10);
var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("drop1", Dropout(0.1)), ("lin2", lin2));

var x = torch.randn(64, 1000);
var y = torch.randn(64, 10);

var optimizer = torch.optim.Adam(seq.parameters());
var loss = functional.mse_loss(Reduction.Sum);

for (int i = 0; i < 10; i++) {
    var eval = seq.forward(x);
    var output = loss(eval, y);

    optimizer.zero_grad();

    output.backward();

    optimizer.step();
}
```

## A Few Things to Know

While the intent has been to stay close to the Pytorch experience, there are some peculiarities to take note of:

1. We have disregarded .NET naming conventions in favor of Python where it impacts the experience. We know this will feel wrong to some, but after a lot of deliberation, we decided to follow the lead of the SciSharp community and embrace naming similarity with Python over .NET tradition. We believe this will make it easier to take Python-based examples and snippets and apply them in .NET.

2. In order to make a constructor call look more the Pytorch code, each class has a factory method with the same name. Because we cannot have a method and a class with the same name in a scope, we moved the class declarations to a nested scope 'Modules.'

    For example:

    ```csharp

    Module conv1 = Conv1d(...);

    ```
    creates an instance of `Modules.Conv1d`, which has 'torch.Module' as its base class.

3. C# uses ':' when passing a named parameter, while F# and Python uses '=', and Pytorch functions have enough parameters to encourage passing them by name. This means that you cannot simply copy a lot of code into C#.

4. There are a number of APIs where Pytorch encodes what are effectively enum types as strings. We have chosen to use proper .NET enumeration types in most cases.

5. The type `torch.device` is `torch.Device` in TorchSharp. We felt that using all-lowercase for a class type was one step too far. The device object constructors, which is what you use most of the time, are still called `device()`


# Memory management

See [docfx/articles/memory.md](docfx/articles/memory.md).

# Download

TorchSharp is distributed via the NuGet gallery: https://www.nuget.org/packages/TorchSharp/

To use TorchSharp, you also need one of the LibTorch backend packages: https://www.nuget.org/packages?q=libtorch, specifically one of

* `libtorch-cpu-linux-x64` (CPU, Linux)

* `libtorch-cpu-win-x64` (CPU, Windows)

* `libtorch-cpu-osx-x64` (CPU, OSX)

* `libtorch-cpu` (CPU, references all three, larger download but simpler)

* `libtorch-cuda-11.1-linux-x64` (CPU/CUDA 11.1, Linux)

  > NOTE: Due to the presence of very large native binaries, using the `libtorch-cuda-11.1-linux-x64` package requires
  > .NET 6, e.g. .NET SDK version `6.0.100-preview.5.21302.13` or greater.

* `libtorch-cuda-11.1-win-x64` (CPU/CUDA 11.1, Windows)

Alternatively you can access the libtorch native binaries via direct reference to existing local native
binaries of LibTorch installed through other means (for example, by installing [PyTorch](https://pytorch.org/) using a Python package manager). You will have to add an explicit load of the relevant native library, for example:

```csharp
    using System.Runtime.InteropServices;
    NativeLibrary.Load("/home/gunes/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch.so")
```

# Code of Conduct
The .NET Foundation has adopted the Contributor Covenant You can read more guidance on the Code of Conduct here. However, it is good practice to include the following text somewhere appropriate in your README.

This project has adopted the code of conduct defined by the Contributor Covenant to clarify expected behavior in our community.
For more information see the [.NET Foundation Code of Conduct](https://dotnetfoundation.org/code-of-conduct).

# Developing

See [DEVGUIDE.md](DEVGUIDE.md).

# Uses

[DiffSharp](https://github.com/DiffSharp/DiffSharp/) also uses this
repository extensively and has been a major factor in iterating support.
