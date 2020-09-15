# Memory Management

Three approaches are available for memory management.

- If having trouble with GPU memory you may have to resort to technique 1.

- If having trouble with CPU memory you may have to resort to technique 3, else 1.

DiffSharp relied on technique 2 and 3.

## Technique 1. Explicit disposal

   In this technique specific tensors (CPU and GPU) are explicitly disposed
   using `using` in C# or explicit calls to `System.IDisposable.Dispose()`.

   👍 control

   👎 you must know when to dispose

   > NOTE: Disposing a tensor only releases the underlying storage if this is the last
   > live TorchTensor which has a view on that tensor.

## Technique 2. Implicit disposal using finalizers

   In this technique all tensors (CPU and GPU) are implicitly disposed via .NET finalizers.

   👍 Simple

   👎 The .NET GC doesn't know of the memory pressure from CPU tensors, so failure may happen if large tensors can't be allocated

   👎 The .NET GC doesn't know of GPU resources

## Technique 3. Implicit disposal using finalizers with memory pressure

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


