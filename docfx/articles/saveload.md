# Saving and Restoring Model Weights and Buffers

There are typically two kinds of state in a model -- parameters, which contain trained weights, and buffers, which contain data that is not trained, but still essential for the functioning of the model. Both should generally be saved and loaded when serializing models.

When using PyTorch, the expected pattern to use when saving and later restoring models from disk or other permanent storage media, is to get the model's state and pickle that using the standard Python format, which is what torch.save() does.

```Python
torch.save(model.state_dict(), 'model_weights.pth')
```

When restoring the model, you are expected to first create a model of the exact same structure as the original, with random weights, then restore the state from a unpickled object:

```Python
model = [...]
model.load_state_dict(torch.load('model_weights.pth'))
```

This presents a couple of problems for a .NET implementation. 

Python pickling is intimately coupled to Python and its runtime object model. It is a complex format that supports object graphs forming DAGs, faithfully maintaining all object state in the way necessary to restore the Python object later.

In order to share models between .NET applications, Python pickling is not at all necessary, and even for moving model state from Python to .NET, it is overkill. The state of a model is a simple dictionary where the keys are strings and the values are tensors.

Therefore, TorchSharp, in its current form, implements its own very simple model serialization format, which allows models originating in either .NET or Python to be loaded using .NET, as long as the model was saved using the special format.

The MNIST and AdversarialExampleGeneration examples in this repo rely on saving and restoring model state -- the latter example relies on a pre-trained model from MNST.

><br/>A future version of TorchSharp may include support for reading and writing Python pickle files directly.<br/><br/>

## How to use the TorchSharp format

In C#, saving a model looks like this:

```C#
model.save("model_weights.dat");
```

And loading it again is done by:

```C#
model = [...];
model.load("model_weights.dat");
```

For efficient memory management, the model should be created on the CPU before loading weights, then moved to the target device. 

><br/>It is __critical__ that all submodules and buffers in a custom module or composed by a Sequential object have exactly the same name in the original and target models, since that is how persisted tensors are associated with the model into which they are loaded.<br/><br/>The CustomModule 'RegisterComponents' will automatically find all fields that are either modules or tensors, register the former as modules, and the latter as buffers. It registers all of these using the name of the field, just like the PyTorch Module base class does.<br/><br/>

### TorchSharp.PyBridge: An Alternate Approach

For those seeking additional flexibility, especially in a mixed .NET and Python environment, TorchSharp.PyBridge offers an alternative approach. Developed by Shaltiel Shmidman, this extension library facilitates seamless interoperability between .NET and Python for model serialization, simplifying the process of saving and loading PyTorch models in a .NET environment.

Key features include:

- `load_py` Method: Easily load PyTorch models saved in the standard Python format directly into TorchSharp.

- `save_py` Method: Save TorchSharp models in a format that can be directly loaded in PyTorch, offering cross-platform model compatibility.

Please note that TorchSharp.PyBridge is not maintained by the TorchSharp team and is an independent extension package. For detailed usage instructions, limitations, and more information, visit [TorchSharp.PyBridge on GitHub](https://github.com/shaltielshmid/TorchSharp.PyBridge).

### Saving a TorchSharp format model in Python

If the model starts out in Python, there's a simple script that allows you to use code that is very similar to the Pytorch API to save models to the TorchSharp format. Rather than placing this trivial script in a Python package and publishing it, we choose to just refer you to the script file itself, [exportsd.py](../../src/Python/exportsd.py), which has all the necessary code.

```Python
f = open("model_weights.dat", "wb")
exportsd.save_state_dict(model.to("cpu").state_dict(), f)
f.close()
```

### Loading a TorchSharp format model in Python

If the model starts out in TorchSharp, there's also a simple script that allows you to load TorchSharp models in Python. All the necessary code can be found in [importsd.py](../../src/Python/importsd.py). And there is an example for using the script:

```Python
f = open("model_weights.dat", "rb")
model.load_state_dict(importsd.load_state_dict(f))
f.close()
```

Also, you can check [TestSaveSD.cs](../../test/TorchSharpTest/TestSaveSD.cs) and [pyimporttest.py](../../test/TorchSharpTest/pyimporttest.py) for more examples.
