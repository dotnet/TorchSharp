## TorchSharp Release Notes

Releases, starting with 9/2/2021, are listed with the most recent release at the top.

### NuGet Version 0.93.7

__Fixed Bugs:__

#413 Random Distributions Should Take a Generator Argument

__API Changes:__

Added Module.Create<T>() to create a model and load weights.

### NuGet Version 0.93.6

__Fixed Bugs:__

#407 rand() and randn() must check that the data type is floating-point.
#410 Support for passing random number generators to rand(), randn(), and randint()


__API Changes:__

Added some overloads to make F# usage more convenient.
Added convenience overloads to a number of random distribution factories.
Added '_' to the torch.nn.init functions. They overwrite the input tensor, so they should have the in-place indicator.

### NuGet Version 0.93.5

__Fixed Bugs:__

#399 Data<T>() returns span that must be indexed using strides. 

This was a major bug, affecting any code that pulled data out of a tensor view.

__API Changes:__

Tensor.Data<T>() -> Tensor.data<T>()
Tensor.DataItem<T>() -> Tensor.item<T>()
Tensor.Bytes() -> Tensor.bytes
Tensor.SetBytes() -> Tensor.bytes

### NuGet Version 0.93.4

This release introduces a couple of new NuGet packages, which bundle the native libraries that you need:

TorchSharp-cpu
TorchSharp-cuda-linux
TorchSharp-cuda-windows

### NuGet Version 0.93.1

With this release, the native libtorch package version was updated to 1.9.0.11, and that required rebuilding this package.

### NuGet Version 0.93.0

With this release, releases will have explicit control over the patch version number.

__Fixed Bugs:__

Fixed incorrectly implemented Module APIs related to parameter / module registration.
Changed Module.state_dict() and Module.load() to 'virtual,' so that saving and restoring state may be customized.
#353 Missing torch.minimum (with an alternative raising exception)
#327 Tensor.Data<T> should do a type check
#358 Implement ModuleList / ModuleDict / Parameter / ParameterList / ParameterDict

__API Changes:__

Removed the type-named tensor factories, such as 'Int32Tensor.rand(),' etc.

__Documentation Changes:__

Added an article on creating custom modules.

### NuGet Version 0.92.52220

This was the first release since moving TorchSharp to the .NET Foundation organization. Most of the new functionality is related to continuing the API changes that were started in the previous release, and fixing some bugs.

__Fixed Bugs:__

#318 A few inconsistencies with the new naming

__Added Features:__

```
torch.nn.MultiHeadAttention
torch.linalg.cond
torch.linalg.cholesky_ex
torch.linalg.inv_ex
torch.amax/amin
torch.matrix_exp
torch.distributions.*   (about half the namespace)
```

__API Changes:__

CustomModule removed, its APIs moved to Module.
