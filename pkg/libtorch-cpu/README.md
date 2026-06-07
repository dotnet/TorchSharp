# libtorch-cpu

Meta-package that references the platform-specific LibTorch CPU native library packages for all supported platforms (Windows x64/ARM64, Linux x64, macOS ARM64).

This is the CPU-only redistribution of [PyTorch LibTorch](https://pytorch.org/) native binaries, packaged for use with [TorchSharp](https://www.nuget.org/packages/TorchSharp).

## Installation

Most users should install [TorchSharp-cpu](https://www.nuget.org/packages/TorchSharp-cpu) instead, which bundles both TorchSharp and this package.

```shell
dotnet add package libtorch-cpu
```

## Documentation

- [GitHub Repository](https://github.com/dotnet/TorchSharp)
- [PyTorch](https://pytorch.org/)

## License

LibTorch is redistributed under its own [license terms](https://github.com/dotnet/TorchSharp/blob/main/THIRD-PARTY-NOTICES.txt).
