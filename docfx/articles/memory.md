# Memory Management

Two approaches are available for memory management. Technique 1 is the default and simplest way to program.

- If having trouble with CPU memory you may have to resort to technique 2.

- If having trouble with GPU memory you may have to resort to technique 2.

Note DiffSharp (which uses TorchSharp) relies on techniques 1.

> Most of the examples included will use technique #1, doing frequent explicit calls to GC.Collect() in the training code -- if not after each batch in the training loop, at least after each epoch.

## Technique 1. Implicit disposal using finalizers

In this technique all tensors (CPU and GPU) are implicitly disposed via .NET finalizers.

When allocation of tensors via `Float32Tensor.*` (likewise `Int64Tensor.*` etc.) fails (whether on GPU or CPU),
then TorchSharp forces a .NET garbage collection and execution of all pending finalizers.

This is not yet done when using general tensor operations.  It is possible a more general retry-after-GC-on-out-of-memory will be added at some point.

👍 Simple

👎 The .NET GC doesn't know of the memory pressure from CPU tensors, so failure may happen if large tensors can't be allocated

👎 The .NET GC doesn't know of GPU resources.

👎 Native operations that allocate temporaries, whether on CPU or GPU, may fail -- the GC scheme implemented by TorchSharp only works when the allocation is initiated by .NET code.

## Technique 2. Explicit disposal

In this technique specific tensors (CPU and GPU) are explicitly disposed
using `using` in C# or explicit calls to `System.IDisposable.Dispose()`.

👍 Specific lifetime management of all resources.

👎 Cumbersome, requiring lots of using statements in your code.

👎 You must know when to dispose.

👎 Temporaries are not covered by this approach, so to maximize the benefit, you may have to store all temporaries to variables and dispose.

> NOTE: Disposing a tensor only releases the underlying storage if this is the last
> live TorchTensor which has a view on that tensor -- the native runtime does reference counting of tensors.


## Links and resources

These articles might give you ides about techniques to use to analyse memory. The code is in python but generally will translate across:

* https://gitmemory.com/issue/pytorch/pytorch/31252/565550016

* https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741

* https://discuss.pytorch.org/t/i-run-out-of-memory-after-a-certain-amount-of-batches-when-training-a-resnet18/1911

