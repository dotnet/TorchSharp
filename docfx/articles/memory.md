

# Memory Management

Three approches are available for memory management.

- If having trouble with GPU memory you may have to resort to technique 1.

- If having trouble with CPU memory you may have to resort to technique 3, else 1.

DiffSharp relied on technique 2 and 3.

## 1. Explicit disposal of large tensors using IDisposable

In this technique specific tensors (CPU and GPU) are explicitly disposed
using `using` in C# or explicit calls to `System.IDisposable.Dispose()`.

Pro: control

Con: you must know when to dispose

> NOTE: Disposing a tensor only releases the underlying storage if this is the last
> live TorchTensor which has a view on that tensor.

## 2. Implicit disposal using finalizers

In this technique all tensors (CPU and GPU) are implicitly disposed via .NET finalizers.

Pro: Simple

Con: Failure may happen if large tensors can't be allocated

Con: The .NET GC doesn't know of the memory pressure from CPU tensors

Con: The .NET GC doesn't know of GPU resources

## 3. Implicit disposal using finalizers with memory pressure

In this technique all tensors (CPU and GPU) are implicitly disposed via .NET finalizers,
but you can explicitly register large tensors as giving memory pressure

    var t = <make a large tensor>
    t.RegisterForMemoryPressure()

This helps the .NET GC reliably know when to force GC and finalization of large tensors
(even though they are small .NET objects)

Memory pressure is automatically removed when the tensor is finalised.

You should only register a tensor for memory pressure once, and should **not** register derived
tensors which use the same storage (e.g. views, slices).

Pro: You have to know when to register memory pressure

Con: The .NET GC doesn't know of GPU resources


