# Memory Management

Three approaches are available for memory management. Technique 1 is the default and simplest way to program. It is the recommended starting point.

- If having trouble with CPU memory you may have to resort to technique 2 or 3.

- If having trouble with GPU memory you may have to resort to technique 2 or 3.

In both cases, you may want to experiment with using a smaller batch size -- temporary tensor values produced by computation on the training data is the main memory problem, and they are in most cases proportional to the batch size. A smaller batch size means more batches, which will take longer, but can often speed up training convergence.

Note DiffSharp (which uses TorchSharp) relies on techniques 1.

Also refer to [Memory Leak Troubleshooting](memory leak troubleshooting.md) for help on fixing any leaks.

> Most of the examples included will use technique #1, doing frequent explicit calls to GC.Collect() in the training code -- if not after each batch in the training loop, at least after each epoch.

## Technique 1. Automatic disposal via Garbage Collection

In this technique all tensors (CPU and GPU) are implicitly disposed via .NET finalizers. Just allocate new tensors to your heart's content, and let GC take care of them. It will only work for small models that do not require a lot of memory. If you do use this approach, you may want to place a call to `GC.Collect()` after each mini-batch of data. It is generally not sufficient to do it at the end of each epoch.

👍 Simple

👎 The .NET GC doesn't know about the memory pressure from CPU tensors, so failure may happen if large tensors can't be allocated

👎 The .NET GC doesn't know about GPU resources.

👎 Native operations that allocate temporaries, whether on CPU or GPU, may fail -- the GC scheme implemented by TorchSharp only works when the allocation is initiated by .NET code.


## Technique 2. Explicit disposal via 'using var x = ...' (C#) or 'use x = ...' (F#)

This technique is more cumbersome, but will result in better performance and have a higher memory ceiling. For many non-trivial models, it is more or less required in order to train on a GPU.

👍 Specific lifetime management of all resources.

👎 Cumbersome, requiring lots of using statements in your code.

👎 You must know when to dispose.

👎 Temporaries are not covered by this approach, so to maximize the benefit, you may have to store all temporaries to variables and dispose.

__Note__: Even with this approach, it is a good idea to place a call to `GC.Collect()` after each mini-batch of data. There may be temporaries that were overlooked, or inconvenient to pull out, or ones where the lifetime was unclear; calling `GC.Collect()` will catch them.


### Returning Fresh References

It is important to understand that all TorchSharp "tensors" (type Tensor) are actually "tensor aliases", referring to a C++ tensor. When a C++ tensor is created and returned to .NET as a tensor alias, and the reference count on the C++ tensor is incremented. When you call `Dispose()` on the TorchSharp tensor alias (that is, type Tensor), it is decremented. If the tensor alias is finalized instead, the decrement happens implicitly.

To enable this technique, all operations that return one or more TorchSharp `Tensor`s should return "fresh" Tensor aliases (though that doesn't always mean freshly copied C++ tensors). This is true even for in-place, destructive operations like `add_()`, which overwrites the underlying native tensor with data, but still returns a fresh tensor alias to that same tensor.

Thus, when you write methods and functions that take and produce type Tensor, for example in the `forward()` method of a model, you should always make sure to return a fresh alias. Most of the time, this happens automatically, because the last action of your code will normally be to call another tensor function, which itself will be returning a fresh alias, but there are cases when it's not, especially when returning input tensors or tensors stored in some lookaside table.

For example, consider a function that returns its input if its one-dimensional, otherwise it returns a reshaped version:

```C#
Tensor flatten(Tensor input) {
    if (input.shape.Length == 1)
        return input.alias();
    else
        return input.reshape(input.numel());
}
```

The `alias()` function avoids doing a clone of the tensor, but still returns a fresh tensor. I you simply return `input`, the caller won't know whether both input and output should be disposed, so the protocol is to always return a fresh tensor.

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

Some additional examples, in F# this time:

```fsharp

let myTensorFunction0(input: Tensor) =
    input.alias()

let myTensorFunction1() =
    if today then
       table[4].alias()
    else
       table[5].alias()

let myTensorFunction2(input: Tensor) =
    input.add(tensor(1))

let myTensorFunction3(input: Tensor) =
    use tmp = input.add(tensor(1))
    tmp.add(tensor(1))

let myTensorFunction4(input: Tensor) =
    use tmp1 = input.add(tensor(1))
    use tmp2 = input.add(tensor(1))
    tmp2.add(tensor(1))

let myTensorFunction5(go: bool, input: Tensor) =
    if go then
        use tmp1 = input.add(tensor(1))
        use tmp2 = input.add(tensor(1))
        tmp2.add(tensor(1))
    else
        input.alias()

let myTensorFunction5(go: bool, input: Tensor) =
    if go then
        use tmp1 = input.add_(tensor(1))  // NOTE: even for in-place mutations
        use tmp2 = input.add_(tensor(1))  // NOTE: even for in-place mutations
        tmp2.add(tensor(1))
    else
        input.alias()
```

### Use 'Sequential' when possible.

Rather than passing tensor arguments between neural network layers inside a custom module's `forward()`, you should rely on the 'Sequential' layer collection, which will be efficient at memory management.

It may not be ideal when first experimenting with a model and trying to debug it, but once you are done with that and move on to a full training data set, it is advisable.

## Technique 3: torch.NewDisposeScope()

This approach, which was added in TorchSharp 0.95.4, makes it easier to dispose of tensors, without the non-determinism of technique 1, or the many temporaries of technique 2. It has most of the advantages of technique 2, and the code elegance of technique 1.

👍 Specific lifetime management of all resources, but in groups.

👍 Temporaries __are__ covered by this approach.

👎 You don't have fine-grained control over when each tensor is reclaimed. All tensors created while a scope is in effect are disposed at once.

👎 It's a new code pattern, not widely used with other libraries.

Let's look at an example, similar to the earlier, technique 2 example:

```C#
using (var d = torch.NewDisposeScope()) {

    total_acc += (predicted_labels.argmax(1) == labels).sum().cpu().item<long>();

    ...
}
```

What happens here, is that all tensors that are created while `d` is alive will be disposed when `d` is disposed. This includes temporaries, so you don't have to do anything special to get them to be disposed. (There's a problem with this simplistic example, which will be discussed later.)

In F#, it would look like:

```F#
use d = torch.NewDisposeScope()

total_acc <- total_acc + (predicted_labels.argmax(1) == labels).sum().cpu().item<long>()
```

If you need to dispose some tensors before the scope is disposed, you can use `DisposeEverything()`, or `DisposeEverythingBut(...)` if you want to exclude a few tensors from disposal. These can be useful when tensor lifetimes aren't cleanly nested in dynamic scopes.

__NOTE: It is absolutely essential for the proper functioning of dynamic dispose scopes that the scope is created with a 'using' statemen (C#) or 'use' expression (F#).__

It's important to note that these scopes are dynamic -- if any functions are called, the tensors inside them are also registered and disposed, unless there's a nested scope within those functions.

It is advisable to place a dispose scope around your training and test code, and in any library code that can be called from contexts that do not have dispose scopes.

That said, you should use dispose scope very carefully: having _too few_ scope raises the pressure on native memory, which is particularly bad for GPUs. Having too _many_ scopes, managing too few temporaries, will add runtime overhead to computations. For example, it may be better to put a scope outside an inner loop that contains multiple computations than to place it inside the loop. There is no single best answer.


### Passing Tensors Out of Scopes

Any tensor that needs to survive the dynamic dispose scope must be either removed from management completely, or promoted to a nesting (outer) scope.

For example, if a tensor variable `tensor` is overwritten in a scope, there are two problems:

1. The tensor held in `tensor` will be overwritten. Since it's not created within the scope, the scope will not dispose of it. A nesting scope must be added to managed the lifetime of all tensors kept in `tensor`:

```C#
using (var d0 = torch.NewDisposeScope()) {

    var tensor = torch.zeros(...)

    for ( ... ) {
        ...
        using (var d1 = torch.NewDisposeScope()) {

            var x = ...;
            tensor += x.log();
            ...
        }
    }
}
```

2. The new tensor that is placed in `tensor` will be disposed when the scope is exited. Since the static scope of `tensor` is not the same as the dynamic scope where it is created, there's a problem.

This is probably not what you intended, so the tensor needs to be either detached, or moved to an outer scope (if one exists).

For example:

```C#
using (var d0 = torch.NewDisposeScope()) {

    var tensor = torch.zeros(...)

    for ( ... ) {
        ...
        using (var d1 = torch.NewDisposeScope()) {

            var x = ...;
            tensor = (tensor + x.log()).MoveToOuterDisposeScope();
            ...
        }
    }
}
```

Sometimes, less is more -- a simple solution is to have fewer nested scopes:

```C#
using (var d0 = torch.NewDisposeScope()) {

    var tensor = torch.zeros(...)

    for ( ... ) {
        ...
        var x = ...;
        tensor += x.log();
        ...
    }
}
```

but sometimes, you still have to move the tensor out, for example when you return a tensor from a method:

```C#
public Tensor foo() {
    using (var d0 = torch.NewDisposeScope()) {

        var tensor = torch.zeros(...)

        foreach ( ... ) {
            ...
            var x = ...;
            tensor += x.log();
            ...
        }

        return tensor.MoveToOuterDisposeScope();
    }
}
```

These examples show how to move a tensor up one level in the stack of scopes. To completely remove a tensor from scoped management, use `DetatchFromDisposeScope()` instead of `MoveToOuterDisposeScope()`.

Even with this technique, it is a good practice to use `Sequential` when possible.
<br/>

### torch.WrappedTensorDisposeScope

A conveniece method was added in 0.97.3 -- it is useful for wrapping a complex expression with multiple temporaries without having to set up a scope explicitly.

It is defined as:

```C#
public static Tensor WrappedTensorDisposeScope(Func<Tensor> expr)
{
    using var scope = torch.NewDisposeScope();
    var result = expr();
    return result.MoveToOuterDisposeScope();
}
```

This is particularly useful in one-line functions and properties, such as in this example from the the Pareto distribution class:

```C#
public override Tensor entropy() => torch.WrappedTensorDisposeScope(() => ((scale / alpha).log() + (1 + alpha.reciprocal())));

```

## Data loaders and datasets

Sometimes we'd like to train our model in the following pattern:

```csharp
using var dataLoader = torch.utils.data.DataLoader(...);
for (int epoch = 0; epoch < 100; epoch++)
{
    foreach (var batch in dataLoader)
    {
        using (torch.NewDisposeScope())
        {
            ...
        }
    }
}
```

In this case, you may notice that `batch` (at least the first batch) is created outside the dispose scope, which would cause a potential memory leak.

Of course we could manually dispose them. But actually we don't have to care about that, because the data loader will automatically dispose it before the next iteration.

However this might cause another problem. For example, we will get disposed tensors when using Linq. The behavior could be modified by setting `disposeBatch` to `false`:

```csharp
using TorchSharp;

using var dataset = torch.utils.data.TensorDataset(torch.zeros([3]));

using var dataLoader1 = torch.utils.data.DataLoader(dataset, batchSize: 1);
using var dataLoader2 = torch.utils.data.DataLoader(dataset, batchSize: 1, disposeBatch: false);

Console.WriteLine(dataLoader1.First()[0].IsInvalid); // True
Console.WriteLine(dataLoader2.First()[0].IsInvalid); // False
```

But those tensors would be detached from all the dispose scopes, even if the whole process is wrapped by a scope. (Otherwise it may lead to confusion since the iterations may not happen in the same dispose scope.) So don't forget to dispose them later or manually attach them to a scope. Also, be aware that enumerating the same `IEnumerable` twice could produce different instances:

```csharp
// DON'T DO THIS:
using TorchSharp;

using var dataset = torch.utils.data.TensorDataset(torch.zeros([3]));
using var dataLoader = torch.utils.data.DataLoader(dataset, batchSize: 1, disposeBatch: false);

var tensors = dataLoader.Select(x => x[0]);
DoSomeThing(tensors.ToArray());

foreach (var tensor in tensors)
{
    tensor.Dispose();
    // DON'T DO THIS.
    // The tensor is not the one you have passed into `DoSomeThing`.
}
```

Meanwhile, when writing a dataset on your own, it should be noticed that the data loaders will dispose the tensors created in `GetTensor` after collation. So a dataset like this will not work because the saved tensor will be disposed:

```csharp
using TorchSharp;

using var dataLoader = torch.utils.data.DataLoader(new MyDataset(), batchSize: 1);
foreach (var _ in dataLoader) ;
// System.InvalidOperationException:
// Tensor invalid -- empty handle.

class MyDataset : torch.utils.data.Dataset
{
    private torch.Tensor tensor = torch.zeros([]);
    public override Dictionary<string, torch.Tensor> GetTensor(long index)
    {
        tensor = tensor + 1;
        // The new tensor is attached to the dispose scope in the data loader,
        // and it will be disposed after collation,
        // so in the next iteration it becomes invalid.
        return new() { ["tensor"] = tensor };
    }

    public override long Count => 3;
}
```

Since the actual technique to "catch" the tensors is just a simple dispose scope. So we can write like this to avoid the disposal:

```csharp
class MyDataset : torch.utils.data.Dataset
{
    private torch.Tensor tensor = torch.zeros([]);
    public override Dictionary<string, torch.Tensor> GetTensor(long index)
    {
        var previous = tensor;
        tensor = (previous + 1).DetachFromDisposeScope();
        previous.Dispose(); // Don't forget to dispose the previous one.
        return new() { ["tensor"] = tensor };
    }

    public override long Count => 3;
}
```

Also, if you want a "`Lazy`" collate function, do not directly save the tensors that are passed in. And `DetachFromDisposeScope` does not work in this case because they are kept in another list instead of dispose scopes, due to some multithreading issues. Instead, you could create aliases for them.

## Links and resources

These articles might give you ides about techniques to use to analyse memory. The code is in python but generally will translate across:

* https://gitmemory.com/issue/pytorch/pytorch/31252/565550016

* https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741

* https://discuss.pytorch.org/t/i-run-out-of-memory-after-a-certain-amount-of-batches-when-training-a-resnet18/1911

