// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System.Collections.Generic;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Modified version of VGG to classify CIFAR10 32x32 images.
    /// </summary>
    /// <remarks>
    /// With an unaugmented CIFAR-10 data set, the author of this saw training converge
    /// at roughly 85% accuracy on the test set, after 50 epochs using VGG-16.
    /// </remarks>
    class VGG : Module<Tensor, Tensor>
    {
        // The code here is is loosely based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
        // Licence and copypright notice at: https://github.com/kuangliu/pytorch-cifar/blob/master/LICENSE

        private readonly Dictionary<string, long[]> _channels = new Dictionary<string, long[]>() {
            { "VGG11", new long[] { 64, 0, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0 } },
            { "VGG13", new long[] { 64, 64, 0, 128, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0 } },
            { "VGG16", new long[] { 64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0 } },
            { "VGG19", new long[] { 64, 64, 0, 128, 128, 0, 256, 256, 256, 256, 0, 512, 512, 512, 512, 0, 512, 512, 512, 512, 0 } }
        };

        private readonly Module<Tensor, Tensor> layers;

        public VGG(string name, int numClasses, Device device = null) : base(name)
        {
            var modules = new List<(string, Module<Tensor, Tensor>)>();

            var channels = _channels[name];

            long in_channels = 3;

            for (var i = 0; i < channels.Length; i++) {

                if (channels[i] == 0) {
                    modules.Add(($"MaxPool2d-{i}a", MaxPool2d(kernelSize: 2, stride: 2)));
                } else {
                    modules.Add(($"conv2d-{i}a", Conv2d(in_channels, channels[i], kernelSize: 3, padding: 1)));
                    modules.Add(($"bnrm2d-{i}a", BatchNorm2d(channels[i])));
                    modules.Add(($"relu-{i}b", ReLU(inplace: true)));
                    in_channels = channels[i];
                }
            }
            modules.Add(("avgpool2d", AvgPool2d(kernel_size: 1, stride: 1)));
            modules.Add(("flatten", Flatten()));
            modules.Add(("linear", Linear(512, numClasses)));

            layers = Sequential(modules);

            RegisterComponents();

            if (device != null && device.type == DeviceType.CUDA)
                this.to(device);
        }

        public override Tensor forward(Tensor input)
        {
            return layers.call(input);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) {
                layers.Dispose();
                ClearModules();
            }
            base.Dispose(disposing);
        }
    }
}
