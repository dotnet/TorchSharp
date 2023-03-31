// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Modified version of original AlexNet to fix CIFAR10 32x32 images.
    /// </summary>
    class AlexNet : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> features;
        private readonly Module<Tensor, Tensor> avgPool;
        private readonly Module<Tensor, Tensor> classifier;

        public AlexNet(string name, int numClasses, torch.Device device = null) : base(name)
        {
            features = Sequential(
                ("c1", Conv2d(3, 64, kernelSize: 3, stride: 2, padding: 1)),
                ("r1", ReLU(inplace: true)),
                ("mp1", MaxPool2d(kernelSize: new long[] { 2, 2 })),
                ("c2", Conv2d(64, 192, kernelSize: 3, padding: 1)),
                ("r2", ReLU(inplace: true)),
                ("mp2", MaxPool2d(kernelSize: new long[] { 2, 2 })),
                ("c3", Conv2d(192, 384, kernelSize: 3, padding: 1)),
                ("r3", ReLU(inplace: true)),
                ("c4", Conv2d(384, 256, kernelSize: 3, padding: 1)),
                ("r4", ReLU(inplace: true)),
                ("c5", Conv2d(256, 256, kernelSize: 3, padding: 1)),
                ("r5", ReLU(inplace: true)),
                ("mp3", MaxPool2d(kernelSize: new long[] { 2, 2 })));

            avgPool = AdaptiveAvgPool2d(new long[] { 2, 2 });

            classifier = Sequential(
                ("d1", Dropout()),
                ("l1", Linear(256 * 2 * 2, 4096)),
                ("r1", ReLU(inplace: true)),
                ("d2", Dropout()),
                ("l2", Linear(4096, 4096)),
                ("r3", ReLU(inplace: true)),
                ("d3", Dropout()),
                ("l3", Linear(4096, numClasses))
            );

            RegisterComponents();

            if (device != null && device.type == DeviceType.CUDA)
                this.to(device);
        }

        public override Tensor forward(Tensor input)
        {
            var f = features.call(input);
            var avg = avgPool.call(f);

            using (var x = avg.reshape(new long[] { avg.shape[0], 256 * 2 * 2 }))
                return classifier.call(x);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) {
                features.Dispose();
                avgPool.Dispose();
                classifier.Dispose();
                ClearModules();
            }
            base.Dispose(disposing);
        }
    }

}
