# Memory Management

Three approaches are available for memory management. Technique 1 is the default and simplest way to program.

- If having trouble with CPU memory you may have to resort to technique 2, else 3.

- If having trouble with GPU memory you may have to resort to technique 3.

Note DiffSharp (which uses TorchSharp) relies on techniques 1 (and sometime soon 2).



## Technique 1. Implicit disposal using finalizers

   In this technique all tensors (CPU and GPU) are implicitly disposed via .NET finalizers.

   👍 Simple

   👎 The .NET GC doesn't know of the memory pressure from CPU tensors, so failure may happen if large tensors can't be allocated

   👎 The .NET GC doesn't know of GPU resources

## Technique 2. Implicit disposal using finalizers with memory pressure

   In this technique all tensors (CPU and GPU) are implicitly disposed via .NET finalizers,
   but you can explicitly register large tensors as giving memory pressure

       var t = <make a large tensor>
       t.RegisterAsMemoryPressure()

   This helps the .NET GC reliably know when to force GC and finalization of large tensors
   (even though they are small .NET objects). Memory pressure is automatically removed when the tensor is finalised.

   You should only register a tensor for memory pressure once, and should **not** register derived
   tensors which use the same storage (e.g. views, slices).

   👍 You have to know when to register memory pressure

   👎 The .NET GC doesn't know of GPU resources


## Technique 3. Explicit disposal

   In this technique specific tensors (CPU and GPU) are explicitly disposed
   using `using` in C# or explicit calls to `System.IDisposable.Dispose()`.

   👍 control

   👎 you must know when to dispose

   > NOTE: Disposing a tensor only releases the underlying storage if this is the last
   > live TorchTensor which has a view on that tensor.

## Re-attempting allocation of large tensors

When allocation of tensors via `FloatTensor.*` (likewise `LongTensor.*` etc.) fails (whether on GPU or CPU),
then TorchSharp forces a .NET garbage collection and execution of all pending finalizers.

This is not yet done when using general tensor operations.  It is possible a more general retry-after-GC-on-out-of-memory will be added at some point.

## Links and resources

These articles might give you ides about techniques to use to analyse memory. The code is in python but generally will translate across:

* https://gitmemory.com/issue/pytorch/pytorch/31252/565550016

* https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741

* https://discuss.pytorch.org/t/i-run-out-of-memory-after-a-certain-amount-of-batches-when-training-a-resnet18/1911



