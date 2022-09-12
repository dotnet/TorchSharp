# Creating Your Own TorchSharp Modules

Unfortunatly, the word 'Module' is one of the most overloaded terms in software. That said, the module concept is central to TorchSharp, which means we have to define its contextual meaning. 

In the context of TorchSharp, it means the same as in PyTorch: the fundamental building block of all models is the 'Module' class. All neural network layers are derived from Module and the way to create a model for training and inference in your code is to create a new Module. Without it, back-propagation will not work.

## Custom Modules

A custom module is created by deriving a subclass from torch.nn.Module. One that is equivalent to the previous example looks like this:

```C#
        private class TestModule1 : Module
        {
            public TestModule1()
                : base("TestModule1")
            {
                lin1 = Linear(100, 10);
                lin2 = Linear(10, 5);
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                using (var x = lin1.forward(input))
                return lin2.forward(x);
            }

            private Module lin1;
            private Module lin2;
        }
```

Note that the field names in the module class correspond to the names that were passed in to the Sequential constructor in the earlier section.

Custom modules should always call `RegisterComponents()` once all submodules and buffers (see discussion later in this document) have been created. This will register the modules and its parameters with the native runtime, which is essential for it to function correctly. For this to work properly, each submodule should have its own private field (**not** property) in the class. The only exception are submodules that do not have trainable weights, such as activation functions like ReLU or tanh.

The `forward()` method contains the computation of the module. It can contain a mix of TorchSharp primitives, layers, as well as any .NET code. Note, however, that only TorchSharp APIs are capable of operating on data residing in CUDA memory. Therefore, if performance is of the essence, expressing all computation in terms of TorchSharp APIs is essential. Non-TorchSharp APIs should be limited to things that aren't related to the tensor data, things like logging, for example.

In PyTorch, the `forward()` method takes an arbitrary number of arguments, of any type, and supports using default arguments. Currently, TorchSharp only defines two versions of `forward()` -- taking one or two tensors, and returning a single tensor. The TorchSharp implementation of Sequential assumes that it is passing only a single tensor along between its layers. Therefore, any model that needs to pass multiple arguments between layers will have to be custom.

Note that the local variable 'x' was declared in a using statement. This is important in order to deallocate the native memory associated with it as soon as it is no longer needed. For that reason, it is important to pull out temporaries like this into local variables, especially when the code is running on a GPU. (More on this at: [Dispose vs. GC in TorchSharp](memory.md)) 

In other words, the following code is less memory efficient, because it delays reclaiming native memory until the next time GC is run:

```C#
        public override Tensor forward(Tensor input)
        {
            return lin2.forward(lin1.forward(input));
        }
```
## Sequential

For the simplest network architecture, which actually covers a surprising breadth, there is a simplified way to create modules -- the 'Sequential' class. This class is created from a list of Modules (i.e. model components). When data is passed to the model, the Sequential instance will invoke each submodule in the order they were passed when the Sequential instance was created, passing the output from each layer to the next. The output of the final submodule will be the output of the Sequential instance.

```C#
var seq = Sequential(("lin1", Linear(100, 10)), ("lin2", Linear(10, 5)));
...
seq.forward(some_data);
```

There is no real performance reason to use Sequential instead of rolling your own custom module, but there is much less code to write. That said, if using a custom module (up next) is your preference, for whatever reason, that's what you should do. It can be useful to create a custom module first, debug it, and then convert to using Sequential.


## Using Sequential Inside A Custome Module

Custom modules are often combined with Sequential, which is especially useful when building long chains of repetitive blocks inside a custom module. In the C# TorchSharp examples, VGG, MobileNet, and ResNet demonstrate this well. 

To illustrate, this is the code for MobileNet from the TorchSharp examples:

```C#
    class MobileNet : Module
    {
        private readonly long[] planes = new long[] { 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024 };
        private readonly long[] strides = new long[] { 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1 };

        private readonly Module layers;

        public MobileNet(string name, int numClasses, Device device = null) : base(name)
        {
            var modules = new List<(string, Module)>();

            modules.Add(("conv2d-first", 
                Conv2d(3, 32, kernelSize: 3, stride: 1, padding: 1, bias: false)));
            modules.Add(("bnrm2d-first", 
                BatchNorm2d(32)));
            modules.Add(("relu-first",   
                ReLU()));
            MakeLayers(modules, 32);
            modules.Add(("avgpool",      
                AvgPool2d(new long[] { 2, 2 })));
            modules.Add(("flatten",      
                Flatten()));
            modules.Add(("linear",       
                Linear(planes[^1], numClasses)));

            layers = Sequential(modules);

            RegisterComponents();
        }

        private void MakeLayers(List<(string, Module)> modules, long in_planes)
        {

            for (var i = 0; i < strides.Length; i++) {
                var out_planes = planes[i];
                var stride = strides[i];

                modules.Add(($"conv2d-{i}a", 
                    Conv2d(in_planes, in_planes, kernelSize: 3, stride: stride, padding: 1, groups: in_planes, bias: false)));
                modules.Add(($"bnrm2d-{i}a", 
                    BatchNorm2d(in_planes)));
                modules.Add(($"relu-{i}a",   
                    ReLU()));
                modules.Add(($"conv2d-{i}b", 
                    Conv2d(in_planes, out_planes, kernelSize: 1L, stride: 1L, padding: 0L, bias: false)));
                modules.Add(($"bnrm2d-{i}b", 
                    BatchNorm2d(out_planes)));
                modules.Add(($"relu-{i}b",   
                    ReLU()));

                in_planes = out_planes;
            }
        }

        public override Tensor forward(Tensor input)
        {
            return layers.forward(input);
        }
    }
```

## ModuleList

In some circumstances, it's useful to define a dynamic number of modules in a custom module. It could be because you want to parameterize the network architecture, or dynamically choose which layers to run, or just that its tedious to define so many fields. This may be addressed by using a ModuleList to contain the submodules. Unlike Sequential, ModuleList itself does not suffice -- its `forward()` method will throw an exception if invoked.

The purpose is simply to provide a list implementation that automatically registers the submodules when components are registered. You have to iterate through the list in the `forward()` method:

```C#
        private class TestModule1 : Module
        {
            public TestModule1()
                : base("TestModule1")
            {
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                for (int i = 0; i < submodules.Count; i++) {      // Using 'for' instead of 'foreach' can be useful for debugging.
                    input = submodules[i].forward(input); 
                }
            }

            private ModuleList submodules = new ModuleList(Linear(100, 10), Linear(10, 5));
        }
```

## ModuleDict

In some circumstances, it's useful to define a dynamic number of modules in a custom module. It could be because you want to parameterize the network architecture, or dynamically choose which layers to run, or just that its tedious to define so many fields. This may be addressed by using a ModuleList to contain the submodules. Unlike Sequential, ModuleList itself does not suffice -- its `forward()` method will throw an exception if invoked.

The purpose is simply to provide a list implementation that automatically registers the submodules when components are registered. You have to iterate through the list in the `forward()` method:

```C#
        private class TestModule1 : Module
        {
            public TestModule1()
                : base("TestModule1")
            {
                dict.Add("lin1", Linear(100, 10));
                dict.Add("lin2", Linear(10, 5));
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                using (var x = submodules["lin1"].forward(input))
                return submodules["lin2"].forward(x);
            }

            private ModuleDict submodules = new ModuleDict();
        }
```

ModuleDict is an ordered dictionary, so you can also iterate through the dictionary as if it were a list. If so, you will get a sequence of tuples with the submodule name and module in each iteration.

## Parameter

Many modules are just compositions of existing modules, but sometimes it will implement a novel algorithm. In this case, the `forward()` method will not just pass tensors from one module to another, but actually use TorchSharp operators and functions to perform arithmetic directly. If this requires training of parameters, those parameters should be declared directly in the module. The Parameter class is a wrapper around tensor; its only purpose is to make sure that it is registered with the native runtime.

For example, a re-implementation of 'Linear' would look something like:

```C#
        private class MyLinear : Module
        {
            public MyLinear(long input_size, long output_size)
                : base("MyLinear")
            {
                weights = Parameter(torch.randn(input_size, output_size));
                bias = Parameter(torch.zeros(output_size));
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                var mm = torch.matmul(input,weights);
                return mm.add_(bias);
            }

            private Parameter weights;
            private Parameter bias;
        }
```

In this case, we're not relying on 'using' in the `forward()` method, because the temporary is reused as the target by the `add_()` function.

Parameter's dirty little secret is that it will clean out the tensor that is given to its constructor. So, `Parameter()` is preferrably used with another tensor factory (such as in the example above), or a cloned tensor.

## ParameterList

Much like ModuleList, ParameterList is a list of Parameter instances, which is automatically registered with the runtime if found as a field of a module instance.

## ParameterDict

Much like ModuleDict, ParameterDict is a dictionary of Parameter instances, which is automatically registered with the runtime if found as a field of a module instance.

## Buffers

Sometimes, a module needs to allocate tensor that are not trainable, i.e. their values are not modified during back-propagation. An example is a random dropout mask. These are referred to as 'buffers' as opposed to 'parameters' and are treated differently by `RegisterComponents()` -- even though they are not trainable, the native runtime still wants to know about them for other purposes, so it is important to declare them in the module.

Each buffer should be declared as a field of type 'Tensor' (not 'Parameter'). This will ensure that the buffer is registered properly when `RegisterComponents()` is called.


## Modules, 'children()' and 'named_children()'

It is sometimes necessary to create a new model from an existing one and discard some of the final layers. The submodules will appear in the 'named_children' list in the same order that they are declared within the module itself, and when constructing a model based on the children, the layers may be reordered unless the submodules are declared in the same order that they are meant to be invoked.

So, for example:

```C#
        private class TestModule1 : Module
        {
            public TestModule1()
                : base("TestModule1")
            {
                lin1 = Linear(100, 10);
                lin2 = Linear(10, 5);
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                using (var x = lin1.forward(input))
                return lin2.forward(x);
            }

            // Correct -- the layers are declared in the same order they are invoked.
            private Module lin1;
            private Module lin2;
        }

        private class TestModule2 : Module
        {
            public TestModule2()
                : base("TestModule2")
            {
                lin1 = Linear(100, 10);
                lin2 = Linear(10, 5);
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                using (var x = lin1.forward(input))
                return lin2.forward(x);
            }

            // Incorrect -- the layers are not declared in the same order they are invoked.
            private Module lin2;
            private Module lin1;
        }

        ...
        TestModule1 mod1 = ...
        TestModule2 mod2 = ...
        var seq1 = nn.Sequential(mod1.named_children());
        seq1.forward(t);                 // Does the same as mod1.forward(t)
        var seq2 = nn.Sequential(mod2.named_children());
        seq2.forward(t);                 // This probably blows up.
```


