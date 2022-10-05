# TorchScript

TorchSchript is a PyTorch technology that lets you save a subset of PyTorch-based Python code without a dependency on a Python runtime. Such models can be loaded into native code, and therefore into .NET code.

TorchScript is very powerful, because it allows you to save the logic and the weights of a model together, and it furthermore allows the module to be loaded into another program, __without any dependencies on the Python runtime.__ Thus, you can load a model that has been serialized using TorchScript and have it behave as any TorchScript module -- you can use it for training, or you can use it for inference.

## Loading TorchScript Modules

Starting with release 0.96.9, you can load TorchScript modules and functions that have been either traced or scripted in Pytorch. It is, however, not yet possible to create a TorchScript module from scratch using TorchSharp. 

In Python, a TorchScript module can be a class derived from 'nn.Module,' or a function that operates on tensors and stays within the constraints that TorchScript places on it. The script can be formed by tracing or by compiling the code. Refer to the [Pytorch JIT](https://pytorch.org/docs/stable/jit.html) docs for information on the details.

For example, the following Python code creates a TorchScript module. Note the use of type annotations

```Python
from typing import Tuple
import torch
from torch import nn
from torch import Tensor

class MyModule(nn.Module):
  def __init__(self):
    super().__init__()
    self.p = nn.Parameter(torch.rand(10))

  def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    return x + y, x - y
  
  @torch.jit.export
  def predict(self, x: Tensor) -> Tensor:
    return x + self.p

  @torch.jit.export
  def add_scalar(self, x: Tensor, i: int) -> Tensor:
    return x + i

m = MyModule()

m = torch.jit.script(m)
m.save("exported.method.dat")
```

The following can also be used to create a TorchScript script:

```Python
@torch.jit.script
def a(x: Tensor, y: Tensor):
    return x + y, x - y
```

Once you have a TorchScript file, you can load it into TorchSharp using:

```C#
var m = torch.jit.load("file-name");
```

It returns a ScriptModule, which behaves just like any other TorchSharp module. Whether the original script came from a module or a function, it is deserialized as a module. You can use it for training of inference by calling either `train()` or `eval()`. ScriptModules always start out on the CPU, so you have to call `cuda()` in order to move it to a GPU.

Note that if you used __tracing__ to create the TorchScript file in Pytorch, submodules that behave differently in training and eval modes will behave according to the mode they were traced in.

If you use the script module to train, you may want / need to save it afterwards. 

That is easily done using `save()`:

```C#
torch.jit.save(m, "file-name");
```

There's also a type-safe version of `load()` that returns a script module that implements IModule<T1...,TResult>. For the above example(s) `torch.jit.load<Tensor,Tensor,(Tensor, Tensor)>` is the appropriate one to use:

```C#
var m = torch.jit.load<Tensor,Tensor,Tensor>("file-name");
```

While it is possible to save a modified ScriptModule from TorchSharp, it is not (yet) possible to create one _from scratch_ using either tracing or scripting.

Script modules have a main `forward()` method, even if they were created from a function rather than a Module class. However, if the module was created from a class, and other methods have the 'torch.jit.export' annotation, then those methods are also callable from TorchSharp:

```C#
var x = torch.rand(10);
var y = torch.rand(10);
var t0 = m.forward(x,y);
var t1 = m.invoke("predict", x);
var t2 = m.invoke("add_scalar", x, 3.14);
```

Note how methods other than `forward()` must be invoked with their name passed as a string. It may be useful to wrap such modules in a hand-written class, like:

```C#
class TestScriptModule : Module<Tensor, Tensor, (Tensor, Tensor)>
{
    internal TestScriptModule(string filename) : base(nameof(TestScriptModule))
    {
        m = torch.jit.load<(Tensor, Tensor)> (filename);
    }

    public override (Tensor, Tensor) forward(Tensor input1, Tensor input2)
    {
        return m.forward(input1, input2);
    }

    public Tensor predict(Tensor input)
    {
        return m.invoke<Tensor>("predict", input);
    }

    public Tensor add_scalar(Tensor input, int i)
    {
        return m.invoke<Tensor>("add_scalar", input, i);
    }

    private torch.jit.ScriptModule<Tensor, Tensor, (Tensor, Tensor)> m;
}
```

## Compiling TorchScript from text

Besides the ability to load TorchScript modules from files, TorchSharp also offers the ability to create them from Python source code directly using the `torch.jit.compile()` method. This results not in a ScriptModule, but a CompilationUnit, which can contain methods, classes, etc. 

Right now, TorchSharp only offers support for using methods compiled from source:

```C#
string script = @"
  def relu_script(a, b):
    return torch.relu(a + b)
  def add_i(x: Tensor, i: int) -> Tensor:
    return x + i
";

using var cu = torch.jit.compile(script);

var x = torch.randn(3, 4);
var y = torch.randn(3, 4);

var z = (Tensor)cu.invoke("relu_script", x, y); // Return type is 'object'
z = cu.invoke<Tensor>("add_i", x, 1);           // Type-safe return type.
```
