## TorchSharp Release Notes

Releases, starting with 9/2/2021, are listed with the most recent release at the top.
## NuGet Version 0.95.4

__API Changes:__

Added DisposeScopeManager and torch.NewDisposeScope() to facilitate a new solution for managing disposing of  tensors with fewer usings.

__Fixed Bugs:__


### NuGet Version 0.95.3

__API Changes:__

The previously unused Tensor.free() method was renamed 'DecoupleFromNativeHandle()' and is meant to be used in native interop scenarios.
Tensor.Handle will now validate that the internal handle is not 'Zero', and throw an exception when it is. This will catch situations where a disposed tensor is accessed.

__Fixed Bugs:__

There were a number of functions in torchvision, as well as a number of optimizers, that did not properly dispose of temporary and intermediate tensor values, leading to "memory leaks" in the absence of explicit GC.Collect() calls.
A couple of randint() overloads caused infinite recursion, crashing the process.

### NuGet Version 0.95.2

__API Changes:__

Added a Sequential factory method to create Sequential from a list of anonymous submodules.
Added TotalCount and PeakCount static properties to Tensor, useful for diagnostic purposes.

__Fixed Bugs:__

#432 Sequential does not dispose of intermediary tensors.

### NuGet Version 0.95.1

This version integrates with LibTorch 1.10.0.

__API Changes:__

Added a 'strict' option to Module.load().

See tracking issue #416 for a list of new 1.10.0 APIs.
https://github.com/dotnet/TorchSharp/issues/416

### NuGet Version 0.93.9

__Fixed Bugs:__

#414 LRScheduler -- not calling the optimizer to step() [The original, closing fix was actually incorrect, but was then fixed again.]

__API Changes:__

Added the NAdam and RAdam optimizers.
Added several missing and new learning rate schedulers.


### NuGet Version 0.93.8

__Fixed Bugs:__

#413 Random Distributions Should Take a Generator Argument
#414 LRScheduler -- not calling the optimizer to step()

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
