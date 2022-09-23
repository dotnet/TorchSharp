# Loading TorchScript Modules

Starting with release 0.96.9, you can load TorchScript modules and functions that have been either traced or scripted in Pytorch. It is, however, not yet possible to create a TorchScript module from scratch using TorchSharp. Refer to the [Pytorch JIT](https://pytorch.org/docs/stable/jit.html) docs for information on how to create such a file.

TorchScript is very powerful, because it allows you to save the logic and the weights of a model together, and it furthermore allows the module to be loaded into another program, __without any dependencies on the Python runtime.__ Thus, you can load a model that has been serialized using TorchScript and have it behave as any TorchScript module -- you can use it for training, or you can use it for inference.

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

While it is possible to save a modified ScriptModule from TorchSharp, it is not (yet) possible to create one _from scratch_ using either tracing or scripting. Another limitation is that the TorchSharp code assumes that the `forward()` function takes only tensors as its arguments and returns a single tensor, a limitation it shares with other TorchSharp modules.

## ScriptModule

`ScriptModule` is what `torch.jit.load()` returns. As stated earlier, it is and behaves like a Module. It encapsulates the logic of the original model. Script modules can be included in other models, used in Sequential, etc.

To use ScriptModule properly, it is important to know that the type-safe `forward()` methods are all implemented in terms of the general `object -> object` forward(), so they are no more efficient than it.

````C#
public override Tensor forward(Tensor t) => (Tensor)forward((object)t);

public override Tensor forward(Tensor x, Tensor y) => (Tensor)forward((x, y));

public override Tensor forward(Tensor x, Tensor y, Tensor z) => (Tensor)forward((x,y,x));
```