# Memory Management

Two approaches are available for memory management. Technique 1 is the default and simplest way to program.

- If having trouble with CPU memory you may have to resort to technique 2.

- If having trouble with GPU memory you may have to resort to technique 2.

Note DiffSharp (which uses TorchSharp) relies on techniques 1.

## Technique 1. Implicit disposal using finalizers

In this technique all tensors (CPU and GPU) are implicitly disposed via .NET finalizers.

When allocation of tensors via `Float32Tensor.*` (likewise `Int64Tensor.*` etc.) fails (whether on GPU or CPU),
then TorchSharp forces a .NET garbage collection and execution of all pending finalizers.

This is not yet done when using general tensor operations.  It is possible a more general retry-after-GC-on-out-of-memory will be added at some point.

👍 Simple

👎 The .NET GC doesn't know of the memory pressure from CPU tensors, so failure may happen if large tensors can't be allocated

👎 The .NET GC doesn't know of GPU resources

## Technique 2. Explicit disposal

In this technique specific tensors (CPU and GPU) are explicitly disposed
using `using` in C# or explicit calls to `System.IDisposable.Dispose()`.

👍 control

👎 you must know when to dispose

> NOTE: Disposing a tensor only releases the underlying storage if this is the last
> live TorchTensor which has a view on that tensor.

## Links and resources

These articles might give you ides about techniques to use to analyse memory. The code is in python but generally will translate across:

* https://gitmemory.com/issue/pytorch/pytorch/31252/565550016

* https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741

* https://discuss.pytorch.org/t/i-run-out-of-memory-after-a-certain-amount-of-batches-when-training-a-resnet18/1911

