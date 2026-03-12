## TorchSharp Release Notes

Releases, starting with 9/2/2021, are listed with the most recent release at the top.

# NuGet Version 0.106.1

__Build/Infrastructure__:

Upgrade target framework from .NET 6 to .NET 8.<br/>
Suppress CS8981 warning for intentional lowercase type names (e.g. `torch`, `nn`, `optim`) that mirror PyTorch's Python API.<br/>
Update CI pipelines to use .NET 8 SDK.<br/>

# NuGet Version 0.106.0

This release upgrades the libtorch backend to v2.10.0, using CUDA 12.8.

__Bug Fixes__:

#1511 Fix `THSNN_interpolate`.<br/>
#1512 Fix `THSJIT_TensorType_sizes` signature.<br/>
#1504 Fix ABI for `THSTorch_int64_to_scalar`.<br/>
#1508 Remove invalid P/Invoke methods.<br/>
#1518 Remove duplicate declarations in `THSTensor.h`.<br/>

__API Changes__:

#1503 Add ReadOnlySpan overloads to many methods.<br/>
#1478 Fix `torch.jit.ScriptModule.zero_grad`.<br/>
#1495 Make `torchvision.io.read_image` and `torchvision.io.read_image_async` allow subsequent opening of the file for reading.<br/>
#1505 Add `THSTorch_scalar_to_bfloat16`.<br/>
#1507 Improve Scalar to Complex performance.<br/>

__Build/Infrastructure__:

#1510 Support VS 2026 in `build.cmd`.<br/>

# NuGet Version 0.105.2

This release upgrades the libtorch backend to v2.7.1, using CUDA 12.8.

# NuGet Version 0.105.1

__Bug Fixes__:

#1426 Sequential.eval() does not put model into eval mode<br/>
`torch.optim.lr_scheduler.LinearLR` `end_factor` default has been corrected, is now 1.0.<br/>
Update package version of SixLabors.ImageSharp to avoid security vulnerability<br/>
Updated dll dependencies loading to avoid using hardcoded version strings<br/>

__API Changes__:

#1374 Add accumulate to index_put_<br/>
`torch.optim.lr_scheduler.PolynomialLR` `power` type has been corrected, is now double.<br/>
Returning an input tensor has been corrected, is now `alias()`.<br/>
Add `torchvision.transforms.Resize` `interpolation` and `antialias`.<br />

# NuGet Version 0.105.0

Move to libtorch 2.5.1. As with the 2.4.0 release, MacOS / Intel is no longer supported by libtorch, so TorchSharp doesn, either.

# NuGet Version 0.104.0

This is a big change in implementation, but not as big in API surface area. Many of the builtin modules, but not all, were re-implemented in managed code calling into native code via the functional APIs. This has several advantages:

1. Align with the Pytorch implementations.<br/>
2. More easily expose module attributes as properties as Pytorch does.<br/>
3. In some cases, avoid native code altogether.<br/>
4. The builtin modules can serve as "best practice" examples for custom module authors.<br/>

__Breaking Changes__:

The names of several arguments have been changed to align better with Pytorch naming. This may break code that passes such arguments by name, but will be caught at compile time.<br/>
The argument defaults for `torch.diagonal()` and `Tensor.diagonal()` arguments have been corrected.<br/>
The default `newLine` for `str`, `jlstr`, `npstr`, `cstr` and `print` have been corrected.<br/>

__Issues fixed__:

#1397 Look into whether parameter creation from a tensor leads to incorrect dispose scope statistics. This bug was discovered during testing of the PR.<br/>
#1210 Attribute omissions.<br/>
#1400 There may be an error in torchvision.transforms.GaussianBlur<br/>
#1402 diagonal() has incorrect default<br/>

__API Changes__:

 #1382: Add support for torch.nn.functional.normalize<br/>

# NuGet Version 0.103.1

__Breaking Changes__:
#1376 `torch.Tensor.backward`'s function signature has been updated to match PyTorch's implementation. Previously, passing `create_graph` or `retain_graph` by position would work like PyTorch's `torch.Tensor.backward`, but not if passing by name (`create_graph`'s value was swapped with `retain_graph`). This has been corrected; however, this means any code that passes `create_graph` or `retain_graph` by name needs to be updated to reflect the intended functionality.<br/>

__Bug Fixes__:

#1383 `torch.linalg.vector_norm`: Make `ord`-argument optional, as specified in docs<br/>
#1385 PackedSequence now participates in the DisposeScope system at the same level as Tensor objects.<br/>
#1387 Attaching tensor to a DisposeScope no longer makes Statistics.DetachedFromScopeCount go negative.<br/>
#1390 DisposeScopeManager.Statistics now includes DisposedOutsideScopeCount and AttachedToScopeCount. ThreadTotalLiveCount is now exact instead of approximate. ToString gives a useful debug string, and documentation is added for how to troubleshoot memory leaks. Also DisposeScopeManager.Statistics.TensorStatistics and DisposeScopeManager.Statistics.PackedSequenceStatistics provide separate metrics for these objects.<br/>
#1392 ToTensor() extension method memory leaks fixed.<br/>
#1414 tensor.print() - Missing default "newLine" Parameter<br/>

# NuGet Version 0.103.0

Move to libtorch 2.4.0.

# NuGet Version 0.102.8

__Bug Fixes__:

#1359 torch.nn.functional.l1_loss computes a criterion with the MSE, not the MAE.<br/>

# NuGet Version 0.102.6

__Breaking Changes__:

When creating a tensor from a 1-D array, and passing in a shape, there is now an ambiguity between the IList and Memory overloads of `torch.tensor()`. The ambiguity is resolved by removing the `dimensions` argument if it is redundant, or by an explicit cast to IList if it is not.

__API Changes__:

 #1326 Allow arrays used to create tensors to be larger than the tensor. Create tensors from a Memory instance.<br/>

__Bug Fixes__:

#1334 MultivariateNormal.log_prob() exception in TorchSharp but works in pytorch.<br/>

# NuGet Version 0.102.5

__Breaking Changes__:

`torchvision.dataset.MNIST` will try more mirrors. The thrown exception might be changed when it fails to download `MNIST`, `FashionMNIST` or `KMNIST`.<br/>
`ObjectDisposedException` will now be thrown when trying to use a disposed dispose scopes.<br/>
The constructor of dispose scopes is no longer `public`. Use `torch.NewDisposeScope` instead.<br/>

__API Changes__:

#1317 How to set default device type in torchsharp.<br/>
#1314 Grant read-only access to DataLoader attributes<br/>
#1313 Add 'non_blocking' argument to tensor and module 'to()' signatures.<br/>
#1291 `Tensor.grad()` and `Tensor.set_grad()` have been replaced by a new property `Tensor.grad`.<br/>
A potential memory leak caused by `set_grad` has been resolved.<br/>
`Include` method of dispose scopes has been removed. Use `Attach` instead.<br/>
Two more `Attach` methods that accepts `IEnumerable<IDisposable>`s and arrays as the parameter have been added into dispose scopes.<br/>
A new property `torch.CurrentDisposeScope` has been added to provide the ability to get the current dispose scope.<br/>
Add module hooks that take no input/output arguments, just the module itself.<br/>

__Bug Fixes__:

#1300 `Adadelta`, `Adam` and `AdamW` will no longer throw `NullReferenceException` when `maximize` is `true` and `grad` is `null`.<br/>
torch.normal` will now correctly return a leaf tensor.<br/>
New options `disposeBatch` and `disposeDataset` have been added into `DataLoader`.<br/>
The default collate functions will now always dispose the intermediate tensors, rather than wait for the next iteration.<br/>

__Bug Fixes__:

`TensorDataset` will now keep the aliases detached from dispose scopes, to avoid the unexpected disposal.<br/>
`DataLoaderEnumerator` has been completely rewritten to resolve the unexpected shuffler disposal, the ignorance of `drop_last`, the incorrect count of worker, and the potential leak cause by multithreading.<br/>
#1303 Allow dispose scopes to be disposed out of LIFO order.<br/>

# NuGet Version 0.102.4

__Breaking Changes__:

Correct `torch.finfo`. (`torch.set_default_dtype`, `Categorical.entropy`, `_CorrCholesky.check`, `Distribution.ClampProbs`, `FisherSnedecor.rsample`, `Gamma.rsample`, `Geometric.rsample`, `distributions.Gumbel`, `Laplace.rsample`, `SigmoidTransform._call` and `SigmoidTransform._inverse` are influenced.)<br/>

__API Changes__:

#1284 make `torch.unique` and `torch.unique_consecutive` public.<br/>

# NuGet Version 0.102.3

__Breaking Changes__:

The 'paddingMode' parameter of convolution has been changed to 'padding_mode', and the 'outputPadding' is now 'output_padding'.

__API Changes__:

#1243 `fuse_conv_bn_weights` and `fuse_linear_bn_weights` are added.<br/>
#1274 ConvTranspose3d does not accept non-uniform kernelSize/stride values<br/>


# NuGet Version 0.102.2

__Bug Fixes__:

#1257 InverseMelScale in NewDisposeScope doesn't dispose tensors<br/>

# NuGet Version 0.102.1

__Breaking Changes__:

The `kernelSize` parameter in the function and class of `AvgPool1D` was renamed to `kernel_size` to match PyTorch naming.
The `stride` parameter in the `torch.nn.functional.avg_pool1d` call now defaults to `kernelSize` instead of 1, to match the PyTorch behavior.


__Bug Fixes__:

`module.load_state_dict()` throws error for in-place operation on a leaf variable that requires grad. <br/>
#1250 cstr and npstr for 0d tensors <br/>
#1249 torch.nn.functional.avg_pool1d is not working correctly<br/>
`module.load()` with streams which don't read the requested # of bytes throws error. <br/>
 #1246 Issue running in notebook on Apple Silicon<br/>

## NuGet Version 0.102.0

This release upgrades the libtorch backend to v2.2.1.

__Breaking Changes__:

The Ubuntu builds are now done on a 22.04 version of the OS. This may (or may not) affect TorchSharp use on earlier versions.<br/>
The default value for the `end_factor` argument in the constructor for `LinearLR` was changed to 1.0 to match PyTorch.<br/>
Any code that checks whether a device is 'CUDA' and does something rather than checking that it isn't 'CPU' will now fail to work, since there is now support for the 'MPS' device on MacOS.<br/>

__API Changes__:

#652: Apple Silicon support .<br/>
#1219: Added support for loading and saving tensors that are >2GB.<br/>

__Bug Fixes__:

Fixed LinearLR scheduler calculation with misplaced parentheses<br/>
Added `get_closed_form_lr` to scheduler to match PyTorch behavior when specifying epoch in `.step()`<br/>

## NuGet Version 0.101.6

__API Changes__:

#1223: Missing prod function torch.prod or a.prod() where a is Tensor<br/>
#1201: How to access the attributes of a model?<br/>
#1094: ScriptModule from Stream / ByteArray<br/>
#1149: Implementation for `torch.autograd.functional.jacobian` to compute Jacobian of a function<br/>
Implemenation for a custom `torch.autograd.Function` class<br/>

__Bug Fixes__:

#1198: CUDA not available when calling backwards before using CUDA<br/>
#1200: Bugs in torch.nn.AvgPool2d and torch.nn.AvgPool3d methods.<br/>

## NuGet Version 0.101.5

__Bug Fixes__:

#1191 : Having trouble moving a module from one GPU to another with gradients.<br/>


## NuGet Version 0.101.4

A fast-follow release addressing a regression in v0.101.3

__Bug Fixes__:

#1185 : Incomplete transfer of module to device (only with 0.101.3)<br/>

## NuGet Version 0.101.3

__Breaking Changes__:

The base `OptimizerState` class was modified and includes two changes:

1. Custom optimizer state objects derived from `OptimizerState` must now explicitly pass the related `torch.nn.Parameter` object to the `OptimizerState` base constructor to maintain correct linkage.
2. Custom state objects must implement an Initialize function. This function is responsible for initializing the properties of the state. Note that this function can be called as a re-intialization, so proper disposal of the previous tensor objects should be handled.

__API Changes__:

Introduced `InferenceMode`, a block-based scoping class for optimizing TorchSharp model inference by disabling gradient computation and enhancing performance.<br/>
Added `Tensor.to_type()` conversion aliases for short, half, bfloat16, cfloat, and cdouble.<br/>
Added `Module.to()` conversion aliases for all the scalar types.<br/>
All distribution classes now implement IDisposable.<br/>

__Bug Fixes__:

#1154 : `mu_product` was not initialized in `NAdam` optimizer<br/>
#1170 : Calling `torch.nn.rnn.utils.pad_packed_sequence` with a CUDA tensor and unsorted_indices threw an error<br/>
#1172 : `optim.LoadStateDict` from an existing `StateDictionary` updated to make sure to copy value and to the right device.<br/>
#1176 : When specific `Optimizers` load in a conditional tensor, made sure to copy to the right device.<br/>
#1174 : Loading CUDA tensor from stream threw an error<br/>
#1179 : Calling `Module.to()` with the `ParameterList` and `ParameterDict` module didn't move the parameters stored in the field.<br/>
#1148 : Calling `Module.to()` shouldn't be differentiable<br/>
#1126 : Calling `ScriptModule.to()` doesn't move attributes<br/>
#1180 : Module.to(ScalarType) has restrictions in PyTorch which aren't restricted in TorchSharp.<br/>

## NuGet Version 0.101.2

__API Changes__:

Added extension method `ScalarType.ElementSize()` to get the size of each element of a given ScalarType.<br/>
Added methods for loading and saving individual tensors with more overloads.<br/>
Added 'persistent' flag to register_buffer()<br/>

__Bug Fixes__:

Fixed byte stream advancement issue in non-strict mode, ensuring proper skipping of non-existent parameters while loading models.<br/>

## NuGet Version 0.101.1

This is a fast-follower bug fix release, addressing persistent issues with stability of using TorchScript from TorchSharp.

__Bug Fixes__:

#1047 Torchscript execution failures (hangs, access violation, Fatal error. Internal CLR fatal error. (0x80131506) )<br/>

## NuGet Version 0.101.0

This is an upgrade to libtorch 2.1.0. It also moves the underlying CUDA support to 12.1 from 11.7, which means that all the libtorch-cuda-* packages have been renamed. Please update your CUDA driver to one that support CUDA 12.1.

__API Changes__:

Enhanced `Module.load` function to return matching status of parameters in non-strict mode via an output dictionary.<br/>
Introduced attribute-based parameter naming for module state dictionaries, allowing custom names to override default field names.<br/>

## NuGet Version 0.100.7

__Breaking Changes__:

DataLoader should no longer be created using `new` -- instead, the overall pattern is followed, placing the classes in `TorchSharp.Modules` and the factories in the static class. This will break any code that creates a DataLoader, but can be fixed by:

1. Removing the `new` in `new torch.utils.data.DataLoader(...)`<br/>
2. Adding a `using TorchSharp.Modules` (C#) or `open TorchSharp.Modules` (F#) to files where `DataLoader` is used as a type name.<br/>

__API Changes__:

Adding an `IterableDataset` abstract class, and making `TensorDataset` derive from it.<br/>
Moving the `DataLoader` class to `TorchSharp.Modules` and adding DataLoader factories.<br/>
#1092: got error when using DataLoader <br/>
#1069: Implementation of torch.sparse_coo_tensor for sparse tensor creation<br/>
Renamed `torch.nn.functional.SiLU` -> `torch.nn.functional.silu`<br/>
Added a set of generic `Sequential` classes.<br/>

__Bug Fixes__:

#1083: Compiler rejects scalar operand due to ambiguous implicit conversion<br/>

## NuGet Version 0.100.6

__Bug Fixes__:

ScriptModule: adding `forward` and the ability to hook.<br/>
Update to SkiaSharp 2.88.6 to avoid the libwebp vulnerability.<br/>
#1105: Dataset files get written to the wrong directory<br/>
#1116: Gradient null for simple calculation<br/>

## NuGet Version 0.100.5

__Breaking Changes__:

Inplace operators no longer create an alias, but instead return 'this'. This change will impact any code that explicitly calls `Dispose` on a tensor after the operation.

__Bug Fixes__:

#1041 Running example code got error in Windows 10<br/>
#1064 Inplace operators create an alias<br/>
#1084 Module.zero_grad() does not work<br/>
#1089 max_pool2d overload creates tensor with incorrect shape<br/>

## NuGet Version 0.100.4

__Breaking Changes__:

The constructor for TensorAccessor is now `internal`, which means that the only way to create one is to use the `data<T>()` method on Tensor. This was always the intent.

__API Changes__:

Tensor.randperm_out() deprecated.<br/>
torch.randperm accepts 'out' argument<br/>
Adding PReLU module.<br/>
Adding scaled_dot_product_attention.<br/>
The constructor for TensorAccessor was made `internal`<br/>
torchvision.utils.save_image implemented<br/>
torchvision.utils.make_grid implemented<br/>
torchvision.transforms.RandAugment implemented<br/>

__Bug Fixes__:

Fixed torch.cuda.synchronize() method<br/>
Suppress runtime warning by setting align_corners to 'false'<br/>
Fixed argument validation bug in Grayscale<br/>
#1056: Access violation with TensorAccessor.ToArray - incompatible data types<br/>
#1057: Memory leak with requires_grad<br/>

## NuGet Version 0.100.3

This release is primarily, but not exclusively, focused on fixing bugs in distributions and adding a few new ones.

__Breaking Changes__:

The two main arguments to `torch.linalg.solve()` and `torch.linalg.solve_ex()` were renamed 'A' and 'B' to align with PyTorch.

__API Changes__:

Adding torch.linalg.solve_triangular()<br/>
Adding torch.distributions.MultivariateNormal<br/>
Adding torch.distributions.NegativeBinomial<br/>
Adding in-place versions of `Tensor.triu()` and `Tensor.tril()`<br/>
Adding torch.linalg.logsigmoid() and torch.nn.LogSigmoid<br/>
A number of distributions were missing the `mode` property.<br/>
Adding a C#-like string formatting style for tensors.<br/>

__Bug Fixes__:

TorchVision `rotate()`, `solarize()` and `invert()` were incorrectly implemented.<br/>
Fixed bug in Bernoulli's `entropy()` and `log_prob()` implementations.<br/>
Fixed bug in Cauchy's `log_prob()` implementation.<br/>
Fixed several bugs in HalfCauchy and HalfNormal.<br/>
The Numpy-style string formatting of tensors was missing commas between elements<br/>

## NuGet Version 0.100.2

__API Changes__:

Add torchvision.datasets.CelebA()<br/>
Add support for properly formatting Tensors in Polyglot notebooks without the 'Register' call that was necessary before.<br/>

__Bug Fixes__:

#1014 AdamW.State.to() ignores returns<br/>
#999 Error in Torchsharp model inference in version 0.100.0<br/>

## NuGet Version 0.100.1

__Breaking Changes__:

TorchSharp no longer supports any .NET Core versions prior to 6.0. .NET FX version support is still the same: 4.7.2 and up.

__API Changes__:

Added operator functionality to Torchvision, but roi are still missing.<br/>
Added support for additional types related to TorchScript modules. Scripts can now return lists of lists and tuples of lists and tuples, to an arbitrary level of nesting.
Scripts can now accept lists of Tensors.

__Bug Fixes__:

#1001 Issue with resnet50, resnet101, and resnet152<br/>

## NuGet Version 0.100.0

Updated backend binaries to libtorch v2.0.1.

Updated the NuGet metadata to use a license expression rather than a reference to a license file. This will help with automated license checking by users.

__Breaking Changes__:

With v2.0.1, `torch.istft()` expects complex numbers in the input tensor.

__API Changes__:

#989 Adding anomaly detection APIs to `torch.autograd`<br/>

__Fixed Bugs__:


## NuGet Version 0.99.6

__Breaking Changes__:

There was a second version of `torch.squeeze()` with incorrect default arguments. It has now been removed.

__API Changes__:

Removed incorrect `torch.squeeze()` method.<br/>
Adding two-tensor versions of `min()` and `max()`<br/>

__Fixed Bugs__:

#984 Conversion from System.Index to TensorIndex is missing<br/>
#987 Different versions of System.Memory between build and package creation.<br/>

## NuGet Version 0.99.5

__API Changes__:

Added Tensorboard support for histograms, images, video, and text.

## NuGet Version 0.99.4

__Breaking Changes__:

There were some changes to the binary format storing optimizer state. This means that any such state generated before updating to this version is invalid and will likely result in a runtime error.

__API Changes__:

Adding torch.tensordot<br/>
Adding torch.nn.Fold and Unfold modules.<br/>
Adding `Module.call()` to all the Module<T...> classes. This wraps `Module.forward()` and allows hooks to be registered. `Module.forward()` is still available, but the most general way to invoke a module's logic is through `call()`.<br/>
Adding tuple overloads for all the padding-related modules.<br/>
Adding support for exporting optimizer state from PyTorch and loading it in TorchSharp<br/>

__Fixed Bugs__:

#842 How to use register_forward_hook?<br/>
#940 Missing torch.searchsorted<br/>
#942 nn.ReplicationPad1d(long[] padding) missing<br/>
#943 LRScheduler.get_last_lr missing<br/>
#951 DataLoader constructor missing drop_last parameter<br/>
#953 TensorDataset is missing<br/>
#962 Seed passed to torch.random.manual_seed(seed) is unused<br/>
#949 Passing optimizer state dictionary from PyTorch to TorchSharp<br/>
#971 std results are inconsistent<br/>

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
