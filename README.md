[![Build Status](https://donsyme.visualstudio.com/TorchSharp/_apis/build/status/xamarin.TorchSharp?branchName=master)](https://donsyme.visualstudio.com/TorchSharp/_build/latest?definitionId=1&branchName=master)

# TorchSharp

TorchSharp is a .NET library that provides access to the library that powers
PyTorch.  

The focus is to bind the API surfaced by libtorch with a particular focus on tensors.

The technology is a "wrapepr library" no more no less. [DiffSharp](https://github.com/DiffSharp/DiffSharp/) uses this
repository extensively and has been a major factor in iterating support.

Things that you can try:

```csharp
using TorchSharp;
using TorchSharp.Tensor;
using TorchSharp.NN;
using static TorchSharp.Tensor.Modules;

var lin1 = Linear(1000, 100);
var lin2 = Linear(100, 10);
var seq = Sequential(lin1, Relu(), lin2);

var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

double learning_rate = 0.00004f;
float prevLoss = float.MaxValue;
var optimizer = Optimizer.Adam(seq.Parameters(), learning_rate);
var loss = Losses.MSE(NN.Reduction.Sum);

for (int i = 0; i < 10; i++)
{
    var eval = seq.Forward(x);
    var output = loss(eval, y);
    var lossVal = output.DataItem<float>();

    Assert.True(lossVal < prevLoss);
    prevLoss = lossVal;

    optimizer.ZeroGrad();

    output.Backward();

    optimizer.Step();
}
```

# Memory management

See [docfx/articles/memory.md](docfx/articles/memory.md).

# Developing

See [DEVGUIDE.md](DEVGUIDE.md).

# Uses

[DiffSharp](https://github.com/DiffSharp/DiffSharp/) also uses this
repository extensively and has been a major factor in iterating support.
