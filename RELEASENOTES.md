## TorchSharp Release Notes

Releases, starting with 9/2/2021, are listed with the most recent release at the top.

## NuGet Version 0.96.8

__Fixed Bugs:__

#618 TorchSharp.Modules.Normal.sample() Expected all tensors [...]<br/>
#621 torch.roll missing<br/>

__API Changes:__

Added torch.from_array() API to create a tensor from an arbitry-dimension .NET array.<br/>
Added torch.tensor() overloads for most common dimensions of .NET arrays: ndim = [1,2,3,4]<br/>
Added the most significant API additions from Pytorch 1.11.
Added two torchaudio APIs.
Added 'decimals' argument to Tensor.round()

## NuGet Version 0.96.7

__Dependency Changes:__

This version integrates with the libtorch 1.11.0 backend. API updates to follow.<br/>

__API Changes:__

Strong name signing of the TorchSharp library to allow loading it in .NET Framework strongly name signed apps.
Added the 'META' device type, which can be used to examine the affect of shape from tensor operations without actually doing any computations.<br/>
Added a few methods from the torch.nn.utils namespace.<br/>
Add torch.stft() and torch.istft()

__Fixed Bugs:__

#567 pad missing the choice to fill at start or end<br/>

## NuGet Version 0.96.6

__API Changes:__

#587 Added the Storage classes, and Tensor.storage()<br/>
Added torchvision.models.resnet***() factories<br/>
Added torchvision.models.alexnet() factory<br/>
Added torchvision.models.vgg*() factories<br/>
Added 'skip' list for loading and saving weights.<br/>
Added torchvision.models.interception_v3() factory<br/>
Added torchvision.models.googlenet() factory<br/>

__Fixed Bugs:__

#582 unbind missing<br/>
#592 GRU and Input and hidden tensors are not at the same device,[...]<br/>
Fixed Module.Dispose() and Sequential.Dispose() (no issue filed)

## NuGet Version 0.96.5

Same-day release. The previous release was made without propert testing of the ToString() improvements in a notebook context. It turned out that when the standard Windows line-terminator "\r\n" is used in a VS Code notebook, an extra blank line is created.

This release fixes that by allowing the caller of ToString() to pass in the line terminator string that should be used when formatting the string. This is easily done in the notebook.

## NuGet Version 0.96.4

In this release, the big change is support for .NET FX 4.7.2 and later.

There are no breaking changes that we are aware of, but see the comment on API Changes below -- backporting code to .NET 4.7 or 4.8, which were not previously supported, may lead to errors in code that uses tensor indexing.

__API Changes:__

Due to the unavailability of `System.Range` in .NET FX 4.7, indexing of tensors using the `[a..b]` syntax is not available. In its place, we have added support for using tuples as index expressions, with the same semantics, except that the "from end" unary operator `^` of the C# range syntax is not available. The tuple syntax is also available for versions of .NET that do support `System.Range`

A second piece of new functionality was to integrate @dayo05's work on DataLoader into the Examples. A couple of MNIST and CIFAR data sets are now found in `torchvision.datasets`

A Numpy-style version of ToString() was added to the existing Julia-style, and the argument to the verbose ToString() was changed from 'Boolean' to an enumeration.

A number of the "bugs" listed below represent missing APIs.

__Fixed Bugs:__

#519 Multiprocessing dataloader support<br/>
#529 pin_memory missing<br/>
#545 Implement FractionalMaxPool{23}d<br/>
#554 Implement MaxUnpool{123}d<br/>
#555 Implement LPPool{12}d<br/>
#556 Implement missing activation modules<br/>
#559 Implement miscellaneous missing layers.<br/>
#564 torch.Tensor.tolist<br/>
#566 Implicit conversion of scalars to tensors<br/>
#576 load_state_dict functionality<br/>

## NuGet Version 0.96.3

__API Changes:__

__NOTE__: This release contains breaking changes.<br/>

The APIs to create optimizers all take 'parameters()' as well as 'named_parameters()' now.<br/>
Support for parameter groups in most optimizers.<br/>
Support for parameter groups in LR schedulers.<br/>

__Fixed Bugs:__

#495 Add support for OptimizerParamGroup<br/>
#509 Tensor.conj() not implemented<br/>
#515 what's reason for making register_module internal?<br/>
#516 AdamW bug on v0.96.0<br/>
#521 Can't set Tensor slice using indexing<br/>
#525 LSTM's forward function not work with null hidden and cell state<br/>
#532 Why does storing module layers in arrays break the learning process?<br/>

## NuGet Version 0.96.2

NOT RELEASED

## NuGet Version 0.96.1

__API Changes:__

__Fixed Bugs:__

Using libtorch CPU packages from F# Interactive required explicit native loads

#510 Module.Load throws Mismatched state_dict sizes exception on BatchNorm1d<br/>

## NuGet Version 0.96.0

__API Changes:__

__NOTE__: This release contains breaking changes.

'Module.named_parameters()', 'parameters()', 'named_modules()', 'named_children()' all return IEnumerable instances instead of arrays.<br/>
Adding weight and bias properties to the RNN modules.<br/>
Lower-cased names: Module.Train --> Module.train and Module.Eval --> Module.eval

__Fixed Bugs:__

#496 Wrong output shape of torch.nn.Conv2d with 2d stride overload<br/>
#499 Setting Linear.weight is not reflected in 'parameters()'<br/>
#500 BatchNorm1d throws exception during eval with batch size of 1<br/>

## NuGet Version 0.95.4

__API Changes:__

Added OneCycleLR and CyclicLR schedulers<br/>
Added DisposeScopeManager and torch.NewDisposeScope() to facilitate a new solution for managing disposing of  tensors with fewer usings.<br/>
Added Tensor.set_()<br/>
Added 'copy' argument to Tensor.to()

__NOTES__: <br/>
The 'Weight' and 'Bias' properties on some modules have been renamed 'weight' and 'bias'.<br/>
The 'LRScheduler.LearningRate' property has been removed. To log the learning rate, get it from the optimizer that is in use.

__Fixed Bugs:__

#476 BatchNorm does not expose bias,weight,running_mean,running_var<br/>
#475 Loading Module that's on CUDA<br/>
#372 Module.save moves Module to CPU<br/>
#468 How to set Conv2d kernel_size=(2,300)<br/>
#450 Smoother disposing

## NuGet Version 0.95.3

__API Changes:__

The previously unused Tensor.free() method was renamed 'DecoupleFromNativeHandle()' and is meant to be used in native interop scenarios.<br/>
Tensor.Handle will now validate that the internal handle is not 'Zero', and throw an exception when it is. This will catch situations where a disposed tensor is accessed.<br/>

__Fixed Bugs:__

There were a number of functions in torchvision, as well as a number of optimizers, that did not properly dispose of temporary and intermediate tensor values, leading to "memory leaks" in the absence of explicit GC.Collect() calls.<br/>
A couple of randint() overloads caused infinite recursion, crashing the process.

## NuGet Version 0.95.2

__API Changes:__

Added a Sequential factory method to create Sequential from a list of anonymous submodules.<br/>
Added TotalCount and PeakCount static properties to Tensor, useful for diagnostic purposes.<br/>

__Fixed Bugs:__

#432 Sequential does not dispose of intermediary tensors.

## NuGet Version 0.95.1

This version integrates with LibTorch 1.10.0.

__API Changes:__

Added a 'strict' option to Module.load().

See tracking issue #416 for a list of new 1.10.0 APIs.
https://github.com/dotnet/TorchSharp/issues/416

## NuGet Version 0.93.9

__Fixed Bugs:__

#414 LRScheduler -- not calling the optimizer to step() [The original, closing fix was actually incorrect, but was then fixed again.]

__API Changes:__

Added the NAdam and RAdam optimizers.<br/>
Added several missing and new learning rate schedulers.


## NuGet Version 0.93.8

__Fixed Bugs:__

#413 Random Distributions Should Take a Generator Argument<br/>
#414 LRScheduler -- not calling the optimizer to step()

__API Changes:__

Added Module.Create<T>() to create a model and load weights.

## NuGet Version 0.93.6

__Fixed Bugs:__

#407 rand() and randn() must check that the data type is floating-point.<br/>
#410 Support for passing random number generators to rand(), randn(), and randint()


__API Changes:__

Added some overloads to make F# usage more convenient.<br/>
Added convenience overloads to a number of random distribution factories.<br/>
Added '_' to the torch.nn.init functions. They overwrite the input tensor, so they should have the in-place indicator.

## NuGet Version 0.93.5

__Fixed Bugs:__

#399 Data<T>() returns span that must be indexed using strides. 

This was a major bug, affecting any code that pulled data out of a tensor view.

__API Changes:__

Tensor.Data<T>() -> Tensor.data<T>()<br/>
Tensor.DataItem<T>() -> Tensor.item<T>()<br/>
Tensor.Bytes() -> Tensor.bytes<br/>
Tensor.SetBytes() -> Tensor.bytes<br/>

## NuGet Version 0.93.4

This release introduces a couple of new NuGet packages, which bundle the native libraries that you need:

TorchSharp-cpu<br/>
TorchSharp-cuda-linux<br/>
TorchSharp-cuda-windows<br/>

## NuGet Version 0.93.1

With this release, the native libtorch package version was updated to 1.9.0.11, and that required rebuilding this package.

## NuGet Version 0.93.0

With this release, releases will have explicit control over the patch version number.

__Fixed Bugs:__

Fixed incorrectly implemented Module APIs related to parameter / module registration.<br/>
Changed Module.state_dict() and Module.load() to 'virtual,' so that saving and restoring state may be customized.<br/>
#353 Missing torch.minimum (with an alternative raising exception)<br/>
#327 Tensor.Data<T> should do a type check<br/>
#358 Implement ModuleList / ModuleDict / Parameter / ParameterList / ParameterDict

__API Changes:__

Removed the type-named tensor factories, such as 'Int32Tensor.rand(),' etc.

__Documentation Changes:__

Added an article on creating custom modules.

## NuGet Version 0.92.52220

This was the first release since moving TorchSharp to the .NET Foundation organization. Most of the new functionality is related to continuing the API changes that were started in the previous release, and fixing some bugs.

__Fixed Bugs:__

#318 A few inconsistencies with the new naming

__Added Features:__

'''
torch.nn.MultiHeadAttention
torch.linalg.cond
torch.linalg.cholesky_ex
torch.linalg.inv_ex
torch.amax/amin
torch.matrix_exp
torch.distributions.*   (about half the namespace)
'''

__API Changes:__

CustomModule removed, its APIs moved to Module.
