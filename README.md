[![Build Status](https://donsyme.visualstudio.com/TorchSharp/_apis/build/status/xamarin.TorchSharp?branchName=master)](https://donsyme.visualstudio.com/TorchSharp/_build/latest?definitionId=1&branchName=master)

# TorchSharp

TorchSharp is a .NET library that provides access to the library that powers
PyTorch.  

The focus is to bind the API surfaced by libtorch with a particular focus on tensors.

The technology is a "wrapper library" no more no less. [DiffSharp](https://github.com/DiffSharp/DiffSharp/) uses this
repository extensively and has been a major factor in iterating support.

Things that you can try:

```csharp
var lin1 = Linear(1000, 100);
var lin2 = Linear(100, 10);
var seq = Sequential(("lin1", lin1), ("relu1", Relu()), ("lin2", lin2));

var x = Float32Tensor.randn(new long[] { 64, 1000 }, deviceIndex: 0, deviceType: DeviceType.CPU);
var y = Float32Tensor.randn(new long[] { 64, 10 }, deviceIndex: 0, deviceType: DeviceType.CPU);

double learning_rate = 0.00004f;
float prevLoss = float.MaxValue;
var optimizer = Optimizer.Adam(seq.parameters(), learning_rate);
var loss = Losses.mse_loss(Reduction.Sum);

for (int i = 0; i < 10; i++)
{
    var eval = seq.forward(x);
    var output = loss(eval, y);
    var lossVal = output.ToSingle();
    Console.WriteLine($"loss = {lossVal}");
    prevLoss = lossVal;

    optimizer.zero_grad();

    output.backward();

    optimizer.step();
}
```

# Memory management

See [docfx/articles/memory.md](docfx/articles/memory.md).

# Developing

See [DEVGUIDE.md](DEVGUIDE.md).

# Uses

[DiffSharp](https://github.com/DiffSharp/DiffSharp/) also uses this
repository extensively and has been a major factor in iterating support.
