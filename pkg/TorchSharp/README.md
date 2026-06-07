# TorchSharp

TorchSharp provides .NET bindings for [PyTorch](https://pytorch.org/), enabling .NET developers to build, train, and deploy deep learning models using the familiar PyTorch API.

## Installation

```shell
dotnet add package TorchSharp
```

You also need a LibTorch runtime package for your platform:

```shell
# CPU only (all platforms)
dotnet add package libtorch-cpu

# CUDA support (Linux)
dotnet add package libtorch-cuda-12.8-linux-x64

# CUDA support (Windows)
dotnet add package libtorch-cuda-12.8-win-x64
```

## Usage

```csharp
using TorchSharp;
using static TorchSharp.torch;

// Create tensors
var x = torch.randn(3, 4);
var y = torch.ones(3, 4);
var z = x + y;

// Build a model
var model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
);
```

## Documentation

- [GitHub Repository](https://github.com/dotnet/TorchSharp)
- [API Documentation](https://dotnet.github.io/TorchSharp/)
- [Developer Guide](https://github.com/dotnet/TorchSharp/blob/main/DEVGUIDE.md)

## License

This project is licensed under the [MIT License](https://github.com/dotnet/TorchSharp/blob/main/LICENSE.txt).
