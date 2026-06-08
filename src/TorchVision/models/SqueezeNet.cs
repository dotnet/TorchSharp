// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class models
        {
            /// <summary>
            /// SqueezeNet 1.0 model from
            /// "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and less than 0.5MB model size".
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last convolutional layer of the classifier will not be loaded from the weights file.</param>
            /// <param name="device">The device to locate the model on.</param>
            /// <remarks>
            /// Pre-trained weights may be retrieved by using Pytorch and saving the model state-dict
            /// using the exportsd.py script, then loading into the .NET instance:
            ///
            /// from torchvision import models
            /// import exportsd
            ///
            /// model = models.squeezenet1_0(pretrained=True)
            /// f = open("model_weights.dat", "wb")
            /// exportsd.save_state_dict(model.state_dict(), f)
            /// f.close()
            ///
            /// See also: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/saveload.md
            ///
            /// In order for the weights to be loaded, the number of classes has to be the same as
            /// in the pre-trained model, which is 1000.
            ///
            /// It is also possible to skip loading the last classifier layer and use it for transfer-learning
            /// with a different number of output classes. To do so, pass skipfc=true.
            ///
            /// All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.SqueezeNet squeezenet1_0(
                int num_classes = 1000,
                float dropout = 0.5f,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return new Modules.SqueezeNet("1_0", num_classes, dropout, weights_file, skipfc, device);
            }

            /// <summary>
            /// SqueezeNet 1.1 model from the official SqueezeNet repo.
            /// SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters than SqueezeNet 1.0.
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last convolutional layer of the classifier will not be loaded from the weights file.</param>
            /// <param name="device">The device to locate the model on.</param>
            /// <remarks>
            /// Pre-trained weights may be retrieved by using Pytorch and saving the model state-dict
            /// using the exportsd.py script, then loading into the .NET instance:
            ///
            /// from torchvision import models
            /// import exportsd
            ///
            /// model = models.squeezenet1_1(pretrained=True)
            /// f = open("model_weights.dat", "wb")
            /// exportsd.save_state_dict(model.state_dict(), f)
            /// f.close()
            ///
            /// See also: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/saveload.md
            ///
            /// In order for the weights to be loaded, the number of classes has to be the same as
            /// in the pre-trained model, which is 1000.
            ///
            /// It is also possible to skip loading the last classifier layer and use it for transfer-learning
            /// with a different number of output classes. To do so, pass skipfc=true.
            ///
            /// All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.SqueezeNet squeezenet1_1(
                int num_classes = 1000,
                float dropout = 0.5f,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return new Modules.SqueezeNet("1_1", num_classes, dropout, weights_file, skipfc, device);
            }
        }
    }

    namespace Modules
    {
        // Based on https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py
        // License: https://github.com/pytorch/vision/blob/main/LICENSE

        public class SqueezeNet : Module<Tensor, Tensor>
        {
            private class Fire : Module<Tensor, Tensor>
            {
                private readonly Module<Tensor, Tensor> squeeze;
                private readonly Module<Tensor, Tensor> squeeze_activation;
                private readonly Module<Tensor, Tensor> expand1x1;
                private readonly Module<Tensor, Tensor> expand1x1_activation;
                private readonly Module<Tensor, Tensor> expand3x3;
                private readonly Module<Tensor, Tensor> expand3x3_activation;

                public Fire(string name, int inplanes, int squeeze_planes, int expand1x1_planes, int expand3x3_planes)
                    : base(name)
                {
                    squeeze = Conv2d(inplanes, squeeze_planes, kernel_size: 1);
                    squeeze_activation = ReLU(inplace: true);
                    expand1x1 = Conv2d(squeeze_planes, expand1x1_planes, kernel_size: 1);
                    expand1x1_activation = ReLU(inplace: true);
                    expand3x3 = Conv2d(squeeze_planes, expand3x3_planes, kernel_size: 3, padding: 1);
                    expand3x3_activation = ReLU(inplace: true);
                    RegisterComponents();
                }

                protected override void Dispose(bool disposing)
                {
                    if (disposing) {
                        squeeze.Dispose();
                        squeeze_activation.Dispose();
                        expand1x1.Dispose();
                        expand1x1_activation.Dispose();
                        expand3x3.Dispose();
                        expand3x3_activation.Dispose();
                    }
                    base.Dispose(disposing);
                }

                public override Tensor forward(Tensor x)
                {
                    x = squeeze_activation.call(squeeze.call(x));
                    return torch.cat(new[] {
                        expand1x1_activation.call(expand1x1.call(x)),
                        expand3x3_activation.call(expand3x3.call(x))
                    }, 1);
                }
            }

            private readonly Module<Tensor, Tensor> features;
            private readonly Module<Tensor, Tensor> classifier;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    features.Dispose();
                    classifier.Dispose();
                }
                base.Dispose(disposing);
            }

            public SqueezeNet(string version, int num_classes = 1000, float dropout = 0.5f,
                string? weights_file = null, bool skipfc = true, Device? device = null)
                : base(nameof(SqueezeNet))
            {
                Module<Tensor, Tensor> final_conv;

                if (version == "1_0") {
                    features = Sequential(
                        Conv2d(3, 96, kernel_size: 7, stride: 2),
                        ReLU(inplace: true),
                        MaxPool2d(kernel_size: 3, stride: 2, ceil_mode: true),
                        new Fire("Fire", 96, 16, 64, 64),
                        new Fire("Fire", 128, 16, 64, 64),
                        new Fire("Fire", 128, 32, 128, 128),
                        MaxPool2d(kernel_size: 3, stride: 2, ceil_mode: true),
                        new Fire("Fire", 256, 32, 128, 128),
                        new Fire("Fire", 256, 48, 192, 192),
                        new Fire("Fire", 384, 48, 192, 192),
                        new Fire("Fire", 384, 64, 256, 256),
                        MaxPool2d(kernel_size: 3, stride: 2, ceil_mode: true),
                        new Fire("Fire", 512, 64, 256, 256)
                    );
                } else if (version == "1_1") {
                    features = Sequential(
                        Conv2d(3, 64, kernel_size: 3, stride: 2),
                        ReLU(inplace: true),
                        MaxPool2d(kernel_size: 3, stride: 2, ceil_mode: true),
                        new Fire("Fire", 64, 16, 64, 64),
                        new Fire("Fire", 128, 16, 64, 64),
                        MaxPool2d(kernel_size: 3, stride: 2, ceil_mode: true),
                        new Fire("Fire", 128, 32, 128, 128),
                        new Fire("Fire", 256, 32, 128, 128),
                        MaxPool2d(kernel_size: 3, stride: 2, ceil_mode: true),
                        new Fire("Fire", 256, 48, 192, 192),
                        new Fire("Fire", 384, 48, 192, 192),
                        new Fire("Fire", 384, 64, 256, 256),
                        new Fire("Fire", 512, 64, 256, 256)
                    );
                } else {
                    throw new ArgumentException($"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected");
                }

                final_conv = Conv2d(512, num_classes, kernel_size: 1);
                classifier = Sequential(
                    Dropout(p: dropout),
                    final_conv,
                    ReLU(inplace: true),
                    AdaptiveAvgPool2d(new long[] { 1, 1 })
                );

                RegisterComponents();

                if (string.IsNullOrEmpty(weights_file)) {
                    foreach (var (_, m) in named_modules()) {
                        if (m is Modules.Conv2d conv) {
                            if (object.ReferenceEquals(m, final_conv)) {
                                nn.init.normal_(conv.weight, mean: 0.0, std: 0.01);
                            } else {
                                nn.init.kaiming_uniform_(conv.weight);
                            }
                            if (conv.bias is not null)
                                nn.init.constant_(conv.bias, 0);
                        }
                    }
                } else {
                    this.load(weights_file!, skip: skipfc ? new[] { "classifier.1.weight", "classifier.1.bias" } : null);
                }

                if (device != null && device.type != DeviceType.CPU)
                    this.to(device);
            }

            public override Tensor forward(Tensor x)
            {
                using (var _ = NewDisposeScope()) {
                    x = features.call(x);
                    x = classifier.call(x);
                    return torch.flatten(x, 1).MoveToOuterDisposeScope();
                }
            }
        }
    }
}
