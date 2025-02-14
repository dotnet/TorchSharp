// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Modified version of ResNet to classify CIFAR10 32x32 images.
    /// </summary>
    class ResNet : Module<Tensor, Tensor>
    {
        // The code here is is loosely based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        // Licence and copypright notice at: https://github.com/kuangliu/pytorch-cifar/blob/master/LICENSE

        private readonly Module<Tensor, Tensor> layers;
        private int in_planes = 64;

        public static ResNet ResNet18(int numClasses, Device device = null)
        {
            return new ResNet(
                "ResNet18",
                (name, in_planes, planes, stride) => new BasicBlock(name, in_planes, planes, stride),
                BasicBlock.expansion, new int[] { 2, 2, 2, 2 },
                10,
                device); 
        }

        public static ResNet ResNet34(int numClasses, Device device = null)
        {
            return new ResNet(
                "ResNet34",
                (name, in_planes, planes, stride) => new BasicBlock(name, in_planes, planes, stride),
                BasicBlock.expansion, new int[] { 3, 4, 6, 3 },
                10,
                device);
        }

        public static ResNet ResNet50(int numClasses, Device device = null)
        {
            return new ResNet(
                "ResNet50",
                (name, in_planes, planes, stride) => new Bottleneck(name, in_planes, planes, stride),
                Bottleneck.expansion, new int[] { 3, 4, 6, 3 },
                10,
                device);
        }

        public static ResNet ResNet101(int numClasses, Device device = null)
        {
            return new ResNet(
                "ResNet101",
                (name, in_planes, planes, stride) => new Bottleneck(name, in_planes, planes, stride),
                Bottleneck.expansion, new int[] { 3, 4, 23, 3 },
                10,
                device);
        }

        public static ResNet ResNet152(int numClasses, Device device = null)
        {
            return new ResNet(
                "ResNet152",
                (name, in_planes, planes, stride) => new Bottleneck(name, in_planes, planes, stride),
                Bottleneck.expansion, new int[] { 3, 4, 36, 3 },
                10,
                device);
        }

        public ResNet(string name, Func<string, int,int,int,Module<Tensor, Tensor>> block, int expansion, IList<int> num_blocks, int numClasses, Device device = null) : base(name)
        {
            var modules = new List<(string, Module<Tensor, Tensor>)>();

            modules.Add(($"conv2d-first", Conv2d(3, 64, kernel_size: 3, stride: 1, padding: 1, bias: false)));
            modules.Add(($"bnrm2d-first", BatchNorm2d(64)));
            modules.Add(($"relu-first", ReLU(inplace:true)));
            MakeLayer(modules, block, expansion, 64, num_blocks[0], 1);
            MakeLayer(modules, block, expansion, 128, num_blocks[1], 2);
            MakeLayer(modules, block, expansion, 256, num_blocks[2], 2);
            MakeLayer(modules, block, expansion, 512, num_blocks[3], 2);
            modules.Add(("avgpool", AvgPool2d(new long[] { 4, 4 })));
            modules.Add(("flatten", Flatten()));
            modules.Add(($"linear", Linear(512 * expansion, numClasses)));

            layers = Sequential(modules);

            RegisterComponents();

            if (device != null && device.type != DeviceType.CPU)
                this.to(device);
        }

        private void MakeLayer(List<(string, Module<Tensor, Tensor>)> modules, Func<string, int, int, int, Module<Tensor, Tensor>> block, int expansion, int planes, int num_blocks, int stride)
        {
            var strides = new List<int>();
            strides.Add(stride);
            for (var i = 0; i < num_blocks-1; i++) { strides.Add(1); }

            for (var i = 0; i < strides.Count; i++) {
                var s = strides[i];
                modules.Add(($"blck-{planes}-{i}", block($"blck-{planes}-{i}", in_planes, planes, s)));
                in_planes = planes * expansion;
            }
        }

        public override Tensor forward(Tensor input)
        {
            return layers.forward(input);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) {
                layers.Dispose();
                ClearModules();
            }
            base.Dispose(disposing);
        }

        class BasicBlock : Module<Tensor, Tensor>
        {
            public BasicBlock (string name, int in_planes, int planes, int stride) : base(name)
            {
                var modules = new List<(string, Module<Tensor, Tensor>)>();

                modules.Add(($"{name}-conv2d-1", Conv2d(in_planes, planes, kernel_size: 3, stride: stride, padding: 1, bias: false)));
                modules.Add(($"{name}-bnrm2d-1", BatchNorm2d(planes)));
                modules.Add(($"{name}-relu-1", ReLU(inplace: true)));
                modules.Add(($"{name}-conv2d-2", Conv2d(planes, planes, kernel_size: 3, stride: 1, padding: 1, bias: false)));
                modules.Add(($"{name}-bnrm2d-2", BatchNorm2d(planes)));

                layers = Sequential(modules);

                if (stride != 1 || in_planes != expansion*planes) {
                    shortcut = Sequential(
                        ($"{name}-conv2d-3", Conv2d(in_planes, expansion * planes, kernel_size: 1, stride: stride, bias: false)),
                        ($"{name}-bnrm2d-3", BatchNorm2d(expansion * planes)));
                }
                else {
                    shortcut = Sequential();
                }

                modules.Add(($"{name}-relu-2", ReLU(inplace: true)));

                RegisterComponents();
            }

            public override Tensor forward(Tensor t)
            {
                var x = layers.forward(t);
                var y = shortcut.forward(t);
                return x.add_(y).relu_();
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    layers.Dispose();
                    shortcut.Dispose();
                    ClearModules();
                }
                base.Dispose(disposing);
            }

            public static int expansion = 1;

            private readonly Module<Tensor, Tensor> layers;
            private readonly Module<Tensor, Tensor> shortcut;
        }

        class Bottleneck : Module<Tensor, Tensor>
        {
            public Bottleneck(string name, int in_planes, int planes, int stride) : base(name)
            {
                var modules = new List<(string, Module<Tensor, Tensor>)>();

                modules.Add(($"{name}-conv2d-1", Conv2d(in_planes, planes, kernel_size: 1, bias: false)));
                modules.Add(($"{name}-bnrm2d-1", BatchNorm2d(planes)));
                modules.Add(($"{name}relu-1", ReLU(inplace:true)));
                modules.Add(($"{name}-conv2d-2", Conv2d(planes, planes, kernel_size: 3, stride: stride, padding: 1, bias: false)));
                modules.Add(($"{name}-bnrm2d-2", BatchNorm2d(planes)));
                modules.Add(($"{name}relu-2", ReLU(inplace: true)));
                modules.Add(($"{name}-conv2d-3", Conv2d(planes, expansion * planes, kernel_size: 1, bias: false)));
                modules.Add(($"{name}-bnrm2d-3", BatchNorm2d(expansion * planes)));

                layers = Sequential(modules);

                if (stride != 1 || in_planes != expansion * planes) {
                    shortcut = Sequential(
                        ($"{name}-conv2d-4", Conv2d(in_planes, expansion * planes, kernel_size: 1, stride: stride, bias: false)),
                        ($"{name}-bnrm2d-4", BatchNorm2d(expansion * planes)));
                } else {
                    shortcut = Sequential();
                }

                RegisterComponents();
            }

            public override Tensor forward(Tensor t)
            {
                var x = layers.forward(t);
                using var y = shortcut.forward(t);
                return x.add_(y).relu_();
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    layers.Dispose();
                    shortcut.Dispose();
                    ClearModules();
                }
                base.Dispose(disposing);
            }

            public static int expansion = 4;

            private readonly Module<Tensor, Tensor> layers;
            private readonly Module<Tensor, Tensor> shortcut;
        }
    }
}
