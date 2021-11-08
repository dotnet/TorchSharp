# Memory Management

Two approaches are available for memory management. Technique 1 is the default and simplest way to program.

- If having trouble with CPU memory you may have to resort to technique 2.

- If having trouble with GPU memory you may have to resort to technique 2.

Note DiffSharp (which uses TorchSharp) relies on techniques 1.

> Most of the examples included will use technique #1, doing frequent explicit calls to GC.Collect() in the training code -- if not after each batch in the training loop, at least after each epoch.

## Technique 1. Automatic disposal via Garbage Collection

In this technique all tensors (CPU and GPU) are implicitly disposed via .NET finalizers. Just allocate new tensors to your heart's content, and let GC take care of them. It will only work for small models that do not require a lot of memory. If you do use this approach, you may want to place a call to `GC.Collect()` after each mini-batch of data. It is generally not sufficient to do it at the end of each epoch.

👍 Simple

👎 The .NET GC doesn't know about the memory pressure from CPU tensors, so failure may happen if large tensors can't be allocated

👎 The .NET GC doesn't know about GPU resources.

👎 Native operations that allocate temporaries, whether on CPU or GPU, may fail -- the GC scheme implemented by TorchSharp only works when the allocation is initiated by .NET code.


## Technique 2. Explicit disposal

This technique is more cumbersome, but will result in better performance and have a higher memory ceiling. For many non-trivial models, it is more or less required in order to train on a GPU.

👍 Specific lifetime management of all resources.

👎 Cumbersome, requiring lots of using statements in your code.

👎 You must know when to dispose.

👎 Temporaries are not covered by this approach, so to maximize the benefit, you may have to store all temporaries to variables and dispose.

__Note__: Even with this approach, it is a good idea to place a call to `GC.Collect()` after each mini-batch of data. There may be temporaries that were overlooked, or inconvenient to pull out, or ones where the lifetime was unclear; calling `GC.Collect()` will catch them.


### Returning Fresh References

It is important to understand that all TorchSharp tensors are simple wrappers around a C++ tensor. The 'Handle' property on `Tensor` is a pointer to a C++ `shared_ptr<at::Tensor>`. When a Tensor is created and returned to .NET, the reference count is incremented. When you call `Dispose()` on the tensor, it is decremented.

To enable this technique, all Tensor operations should return a fresh reference to a Tensor, if not a fresh Tensor. This is true even for in-place, destructive operations like `add_()`, which overwrites the underlying native tensor with data, but still returns a Tensor with a handle that is different from the one going in.

Thus, when you write methods and functions that take and produce Tensor, for example in the `forward()` method of a model, you should always make sure to return a fresh reference. Most of the time, this happens automatically, because your code will be calling runtime code, but there are cases when it's not.

For example, consider a function that returns its input if its one-dimensional, otherwise it returns a reshaped version:

```C#
Tensor flatten(Tensor input) {
    if (input.shape.Length == 1)
        return input.alias();
    else
        return input.reshape(input.numel());        
}
```

The `alias()` function avoids doing a clone of the tensor, but still returns a fresh tensor.

### Disposing Tensor

In order to manage native storage, in particular GPU storage, it is necessary to do some explicit memory management for all temporaries, especially ones that are involved in a model's computation chain.

Here are the simple guidance rules:

1. Create a variable for each computed Tensor.

2. Use the `using` (C#) or `use` (F#) syntax to declare the variable.

3. Don't call Dispose on the input of a function. Let the caller handle its lifetime.

For example, consider this expression from the 'TextClassification' example:

```C#
total_acc += (predicted_labels.argmax(1) == labels).sum().to(torch.CPU).item<long>();
```

There are lots of hidden temporaries in this relatively innocuous expression. In this particular case, it's involved in figuring out whether a prediction was accurate or not, so it's not going to be super-impactful on memory (the tensors are small), but it's still illustrative. A version where all temporaries are pulled out looks like this:

```C#
using var am = predicted_labels.argmax(1);
using var eq = am == labels;
using var sum = eq.sum();
using var moved = sum.to(torch.CPU);

total_acc += moved.item<long>();
```

The most essential places to do explicit memory management is in any function that might be involved with data preparation or the model computation, since the tensors are big and repeatedly used.


### Use 'Sequential' when possible.

Rather than passing tensor arguments between neural network layers inside a custom module's `forward()`, you should rely on the 'Sequential' layer collection, which will be efficient at memory management.

It may not be ideal when first experimenting with a model and trying to debug it, but once you are done with that and move on to a full training data set, it is advisable.

## Links and resources

These articles might give you ides about techniques to use to analyse memory. The code is in python but generally will translate across:

* https://gitmemory.com/issue/pytorch/pytorch/31252/565550016

* https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741

* https://discuss.pytorch.org/t/i-run-out-of-memory-after-a-certain-amount-of-batches-when-training-a-resnet18/1911

