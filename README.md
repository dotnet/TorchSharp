[![Build Status](https://donsyme.visualstudio.com/TorchSharp/_apis/build/status/xamarin.TorchSharp?branchName=master)](https://donsyme.visualstudio.com/TorchSharp/_build/latest?definitionId=1&branchName=master)

# TorchSharp

TorchSharp is a .NET library that provides access to the library that powers
PyTorch.  

The focus is to bind the API surfaced by libtorch with a particular focus on tensors.

The technology is a "wrapper library" no more no less. [DiffSharp](https://github.com/DiffSharp/DiffSharp/) uses this
repository extensively and has been a major factor in iterating support.

Things that you can try:

```csharp
using TorchSharp;

var lin1 = torch.Linear(1000, 100);
var lin2 = torch.Linear(100, 10);
var seq = torch.Sequential(("lin1", lin1), ("relu1", ReLU()), ("drop1", Dropout(0.1)), ("lin2", lin2));

var x = torch.randn(64, 1000);
var y = torch.randn(64, 10);

var optimizer = torch.optim.Adam(seq.parameters());
var loss = torch.nn.functional.mse_loss(NN.Reduction.Sum);

for (int i = 0; i < 10; i++) {
    var eval = seq.forward(x);
    var output = loss(eval, y);

    optimizer.zero_grad();

    output.backward();

    optimizer.step();
}
```

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


# Developing

See [DEVGUIDE.md](DEVGUIDE.md).

# Uses

[DiffSharp](https://github.com/DiffSharp/DiffSharp/) also uses this
repository extensively and has been a major factor in iterating support.
