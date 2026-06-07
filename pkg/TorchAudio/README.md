# TorchAudio

TorchAudio provides .NET bindings for [torchaudio](https://pytorch.org/audio/), enabling audio processing and deep learning for audio tasks in .NET applications. Built on top of [TorchSharp](https://github.com/dotnet/TorchSharp).

## Installation

```shell
dotnet add package TorchAudio
```

You also need the TorchSharp package and a LibTorch runtime package:

```shell
dotnet add package TorchSharp
dotnet add package libtorch-cpu  # or a CUDA variant
```

## Documentation

- [GitHub Repository](https://github.com/dotnet/TorchSharp)
- [TorchSharp API Documentation](https://dotnet.github.io/TorchSharp/)

## License

This project is licensed under the [MIT License](https://github.com/dotnet/TorchSharp/blob/main/LICENSE.txt).
