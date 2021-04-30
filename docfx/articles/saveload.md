# Saving and Restoring Models

When using PyTorch, the expected pattern to use when saving and later restoring models from disk or other permanent storage media, is to get the model's state and pickle that using the standard Python format.

```Python
torch.save(model.state_dict(), 'model_weights.pth')
```

When restoring the model, you are expected to first create a model of the exact same structure as the original, with random weights, then restore the state:

```Python
model = [...]
model.load_state_dict(torch.load('model_weights.pth'))
```

This presents a couple of problems for a .NET implementation. First, Python pickling is very intimately coupled with Python and its runtime object model. It is a complex format that supports object graphs that form DAGs, and faithfully maintaining all object state.

Second, in order to share models between .NET applications, Python pickling is not necessary, and even for moving model state from Python to .NET, it is overkill. The state of a model is a simple dictionary where the keys are strings and the values are tensors.

Therefore, TorchSharp in its current form, implements its own very simple model serialization format, which allows models originating in either .NET or Python to be loaded using .NET, as long as the model was saved using the special format.

The MNIST and AdversarialExampleGeneration examples in this repo rely on saving and restoring model state -- the latter example relies on a pre-trained model from MNST.

> A future version of TorchSharp may include support for reading and writing Python pickle files directly. There are 

## How to use the TorchSharp format


In C#, saving a model looks like this:

```C#
model.save("model_weights.dat");
```

It's important to note that calling 'save' will move the model to the CPU, where it remains after the call. If you need to continue to use the model after saving it, you will have to explicitly move it back:

```C#
model.to(Device.CUDA);
```

And loading it again is done by:

```C#
model = [...];
model.load("model_weights.dat");
```

The model should be created on the CPU before loading weights, then moved to the target device.

If the model starts out in Python, there's a simple script that allows you to use code that is very similar to the Pytorch API to save models to the TorchSharp format. Rather than placing this trivial script in a Python package and publishing it, we choose to just refer you to the script file itself, [exportsd.py](../src/Python/exportsd.py), which has all the necessary code.

```Python
f = open("model_weights.dat", "wb")
exportsd.save_state_dict(model.to("cpu").state_dict(), f)
f.close()
```
