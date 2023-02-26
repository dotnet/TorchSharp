## TorchSharp Release Notes

Releases, starting with 9/2/2021, are listed with the most recent release at the top.

## NuGet Version 0.99.3

__API Changes__:

Adding `Module.call()` to all the Module<T...> classes. This wraps `Module.forward()` and allows hooks to be registered. `Module.forward()` is still available, but the most general way to invoke a module's logic is through `call()`.<br/>

__Fixed Bugs__:

#842 How to use register_forward_hook?


## NuGet Version 0.99.3

__API Changes__:

Fixing misspelling of 'DetachFromDisposeScope,' deprecating the old spelling.<br/>
Adding allow_tf32<br/>
Adding overloads of Module.save() and Module.load() taking a 'Stream' argument.<br/>
Adding torch.softmax() and Tensor.softmax() as aliases for torch.special.softmax()<br/>
Adding torch.from_file()<br/>
Adding a number of missing pointwise Tensor operations.<br/>
Adding select_scatter, diagonal_scatter, and slice_scatter<br/>
Adding torch.set_printoptions<br/>
Adding torch.cartesian_prod, combinations, and cov.<br/>
Adding torch.cdist, diag_embed, rot90, triu_indices, tril_indices<br/>

__Fixed Bugs__:

#913 conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)<br/>
#910 nn.Module.modules is missing<br/>
#912 nn.Module save and state_ dict method error<br/>

## NuGet Version 0.99.2

__API Changes__:

Adding 'maximize' argument to the Adadelta optimizer<br/>
Adding linalg.ldl_factor and linalg.ldl_solve<br/>
Adding a couple of missing APIs (see #872)<br/>
Adding SoftplusTransform<br/>
Support indexing and slicing of Sequential<br/>
Adding ToNDArray() to TensorAccessor<br/>

__Fixed Bugs__:

#870 nn.AvgPool2d(kernel_size=3, stride=2, padding=1) torchsharp not support padding<br/>
#872 Tensor.masked_fill_(mask, value) missing<br/>
#877 duplicate module parameters called named_parameters() while load model by cuda<br/>
#888 THSTensor_meshgrid throws exception<br/>

## NuGet Version 0.99.1

__Breaking Changes__:

The options to the ASGD, Rprop, and RMSprop optimizers have been changed to add a 'maximize' flag. This means that saved state dictionaries for these optimizers will not carry over.

The return type of Sequential.append() has changed from 'void' to 'Sequential.' This breaks binary compatibility, but not source compat.

__API Changes__:

Added a number of 1.13 APIs under `torch.special`<br/>
Added a `maximize` flag to the ASGD, Rprop and RMSprop optimizers.<br/>
Added PolynomialLR scheduler<br/>
The return type of Sequential.append() has changed from 'void' to 'Sequential.'<br/>
Added 1-dimensional array overloads for `torch.as_tensor()`<br/>

__Fixed Bugs__:

#836 Categorical seems to be miscalculated<br/>
#838 New Bernoulli get "Object reference not set to an instance of an object."<br/>
#845 registered buffers are being ignored in move model to device<br/>
#851 tensor.ToString(TorchSharp.TensorStringStyle.Numpy)<br/>
#852 The content returned by torch.nn.Sequential.append() is inconsistent with the official<br/>

## NuGet Version 0.99.0

This is an upgrade to libtorch 1.13. It also moves the underlying CUDA support to 11.7 from 11.3, which means that all the libtorch-cuda-* packages have been renamed.

__Breaking Changes__:

See API Changes.<br/>

__API Changes__:

Removed Tensor.lstsq, paralleling PyTorch. Use torch.linalg.lstsq, instead. This is a breaking change.<br/>
Added 'left' Boolean argument to `torch.linalg.solve()`<br/>

## NuGet Version 0.98.3

__Fixed Bugs__:

MultiStepLR scheduler was not computing the next LR correctly.<br/>
Fixed incorrect version in package reference.<br/>
Added missing package references to TorchVision manifest.<br/>

## NuGet Version 0.98.2

__Breaking Changes__:

The .NET 5.0 is no longer supported. Instead, .NET 6.0 is the minimum version. .NET FX 4.7.2 and higher are still supported.

__API Changes__:

Support 'null' as input and output to/from TorchScript.<br/>
Added support for label smoothing in CrossEntropyLoss.<br/>
Added torchaudio.transforms.MelSpectrogram().<br/>
Adding squeeze_()<br/>
Adding older-style tensor factories -- IntTensor, FloatTensor, etc.<br/>

__Fixed Bugs__:

#783 Download progress bar missing<br/>
#787 torch.where(condition) → tuple of LongTensor function missing<br/>
#799 TorchSharp.csproj refers Skia<br/>

__Source Code Cleanup__:

Moved P/Invoke declarations into dedicated class.<br/>
Added C# language version to all .csproj files.<br/>

## NuGet Version 0.98.1

__Breaking Changes:__

TorchVision and TorchAudio have beeen moved into their own NuGet packages, which need to be added to any project using their APIs.

ModuleList and ModuleDict are now generic types, taking the module type as the type parameter. torch.nn.ModuleDict() will return a ModuleDict<Module>, which torch.nn.ModuleDict<T>() will return a ModuleDict<T>, where T must be a Module type.

__Fixed Bugs:__

#568 Overloads for Named Tensors<br/>
#765 Support invoking ScriptModule methods<br/>
#775 torch.jit.load: support specifying a target device<br/>
#792 Add SkiaSharp-based default imager for torchvision.io<br/>

__API Changes__:

Generic ModuleDict and ModuleList<br/>
Added torchaudio.transforms.GriffinLim
Added support for named tensors
Added default dim argument value for 'cat'

## NuGet Version 0.98.0

__Breaking Changes:__

Some parameter names were changed to align with PyTorch. This affects names like 'dimension,' 'probability,' and 'keepDims' and will break code that is passing these parameters by name.

Module.to(), cpu(), and cuda() were moved to a static class for extension methods. This means that it is necessary to have a 'using TorchSharp;' (C#) or 'open TorchSharp' (F#) in each file using them.

Doing so (rather than qualifying names with 'TorchSharp.') was already recommended as a best practice, since such a using/open directive will allows qualified names to align with the PyTorch module hierarchy.

__Loss functions are now aligned with the PyTorch APIs.__ This is a major change and the reason for incrementing the minor version number. The most direct consequence is that losses are modules rather than delegates, which means you need to call .forward() to actually compute the loss. Also, the factories are in torch.nn rather than torch.nn.functional and have the same Pascal-case names as the corresponding types. The members of the torch.nn.functional static class are now proper immediate loss functions, whereas the previous ones returned a loss delegate.

__Generic Module base class.__ The second major change is that Module is made type-safe with respect to the `forward()` function. Module is now an abstract base class, and interfaces `IModule<T,TResult>`, `IModule<T1,T2,TResult>`,... are introduced to define the signature of the `forward()` function. For most custom modules, this  means that the base class has to be changed to `Module<Tensor,Tensor>`, but some modules may require more significant changes.

ScriptModule follows this pattern, but this version introduces `ScriptModule<T...,TResult>` base classes, with corresponding `torch.jit.load<T...,TResult>()` static factory methods.

__Fixed Bugs:__

#323 forward() should take a variable-length list of arguments<br/>
#558 Fix deviation from the Pytorch loss function/module APIs<br/>
#742 Ease of use: Module.to method should be generic T -> T<br/>
#743 Ease of use: module factories should have dtype and device<br/>
#745 Executing a TorchScript that returns multiple values, throws an exception<br/>
#744 Some of functions with inconsistent argument names<br/>
#749 functional.linear is wrong<br/>
#761 Stateful optimizers should have support for save/load from disk.<br/>
#771 Support more types for ScriptModule<br/>

__API Changes__:

Module.to(), cpu(), and cuda() were redone as extension methods. The virtual methods to override, if necessary, are now named '_to'. A need to do so should be extremely rare.<br/>
Support for saving and restoring hyperparameters and state of optimizers<br/>
Loss functions are now Modules rather than delegates.<br/>
Custom modules should now use generic versions as base classes.<br/>
ScriptModule supports calling methods other than forward()<br/>
Added torch.jit.compile().<br/>

## NuGet Version 0.97.6

__Breaking Changes:__

This release changes TorchSharp.torchvision from a namespace to a static class. This will break any using directives that assumes that it is a namespace.

__Fixed Bugs:__

#719 ResNet maxpool<br/>
#730 Sequential.Add<br/>
#729 Changing torchvision namespace into a static class?<br/>

__API Changes__:

Adding 'append()' to torch.nn.Sequential<br/>
Adding torch.numel() and torch.__version__<br/>
Adding modifiable global default for tensor string formatting<br/>

## NuGet Version 0.97.5

__Fixed Bugs:__

#715 How to implement the following code <br/>

__API Changes__:

Add functional normalizations<br/>
Added torch.utils.tensorboard.SummaryWriter. Support for scalars only.<br/>


## NuGet Version 0.97.3

__Fixed Bugs:__

#694 torch.log10() computes torch.log()<br/>
#691 torch.autograd.backward()<br/>
#686 torch.nn.functional.Dropout() doesn't have the training argument.<br/>

__API Changes__:

Add `repeat_interleave()`<br/>
Add torch.broadcast_shapes()<br/>
Added meshgrid, mT, mH, and H<br/>
Added additional distributions.<br/>
Add dct and mu-law to torchaudio
Added torchvision -- sigmoid_focal_loss()<br/>
Update the arguments of `dropout()` in `Tacotron2`<br/>
Add static function for `all()`, `any()`, `tile()`, `repeat_interleave()`.<br/>
Add an implementation of the ReduceLROnPlateau learning rate scheduler.<br/>

## NuGet Version 0.97.2

__Breaking Changes:__

This release contains a breaking change__ related to `torch.tensor()` and `torch.from_array()` which were not adhering to the semantics of the Pytorch equivalents (`torch.from_numpy()` in the case of `torch.from_array()`).

With this change, there will be a number of different APIs to create a tensor form a .NET array. The most significant difference between them is whether the underlying storage is shared, or whether a copy is made. Depending on the size of the input array, copying can take orders of magnitude more time in creation than sharing storage, which is done in constant time (a few μs).

The resulting tensors may be reshaped, but not resized.

```C#
// Never copy:
public static Tensor from_array(Array input)

// Copy only if dtype or device arguments require it:
public static Tensor frombuffer(Array input, ScalarType dtype, long count = -1, long offset = 0, bool requiresGrad = false, Device? device = null)
public static Tensor as_tensor(Array input,  ScalarType? dtype = null, Device? device = null)
public static Tensor as_tensor(Tensor input, ScalarType? dtype = null, Device? device = null)

// Always copy:
public static Tensor as_tensor(IList<<VARIOUS TYPES>> input,  ScalarType? dtype = null, Device? device = null)
public static Tensor tensor(<<VARIOUS TYPES>> input, torch.Device? device = null, bool requiresGrad = false)
```

__Fixed Bugs:__

#670 Better align methods for creating tensors from .NET arrays with Pytorch APIs. _This is the breaking change mentioned earlier._<br/>
#679 The default value of onesided or torch.istft() is not aligned with PyTorch<br/>

__API Changes__:

Added torch.nn.init.trunc_normal_<br/>
Added index_add, index_copy, index_fill<br/>
Added torch.frombuffer()<br/>
Added torch.fft.hfft2, hfftn, ihfft2, ihfftn<br/>
Adding SequentialLR to the collection of LR schedulers.<br/>
Add 'training' flag to functional dropout methods.<br/>
Add missing functions to torchaudio.functional<br/>
Adding TestOfAttribute to unit tests<br/>

## NuGet Version 0.97.1

This release is made shortly after 0.97.0, since it adresses a serious performance issue when creating large tensors from .NET arrays.

__Fixed Bugs:__

#670 Tensor allocation insanely slow for from_array()<br/>

__API Changes__:

RNN, LSTM, GRU support PackedSequence<br/>
Add element-wise comparison methods of torch class.<br/>
Fix clamp and (non)quantile method declarations<br/>
Implementing isnan()<br/>
Added torchaudio.models.Tacotron2()<br/>

## NuGet Version 0.97.0

__Fixed Bugs:__

#653:Tensor.to(Tensor) doesn't change dtype of Tensor.<br/>

__API Changes__:

Add ability to load and save TorchScript modules created using Pytorch<br/>
Add torch.utils.rnn<br/>
Add torchvision.io<br/>
Add Tensor.trace() and torch.trace() (unrelated to torch.jit.trace)<br/>
Add Tensor.var and Tensor.var_mean<br/>
Add torchaudio.datasets.SPEECHCOMMANDS<br/>
Add torchaudio.Resample()<br/>

## NuGet Version 0.96.8

__Breaking Changes:__

This release contains a fix to inadvertent breaking changes in 0.96.7, related to Tensor.str(). This fix is itself breaking, in that it breaks any code that relies on the order of
arguments to str() introduced in 0.96.7. However, since the pre-0.96.7 argument order makes more sense, we're taking this hit now rather than keeping the inconvenient order in 0.96.7.

__Fixed Bugs:__

#618 TorchSharp.Modules.Normal.sample() Expected all tensors [...]<br/>
#621 torch.roll missing<br/>
#629 Missing dependency in 0.96.7 calling TorchSharp.torchvision.datasets.MNIST<br/>
#632 gaussian_nll_loss doesn't work on GPU<br/>

__API Changes:__

Add torchaudio.datasets.YESNO().<br/>
Added torch.from_array() API to create a tensor from an arbitry-dimension .NET array.<br/>
Added torch.tensor() overloads for most common dimensions of .NET arrays: ndim = [1,2,3,4]<br/>
Added the most significant API additions from Pytorch 1.11.<br/>
Added juliastr() and npstr().<br/>
Added two torchaudio APIs.<br/>
Added 'decimals' argument to Tensor.round()<br/>
Changed tensor.str() to undo the breaking change in 0.96.7<br/>
Added torch.std_mean()<br/>

## NuGet Version 0.96.7

__Dependency Changes:__

This version integrates with the libtorch 1.11.0 backend. API updates to follow.<br/>

__API Changes:__

Strong name signing of the TorchSharp library to allow loading it in .NET Framework strongly name signed apps.<br/>
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
