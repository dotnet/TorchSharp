// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
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
            /// DenseNet-121 model from "Densely Connected Convolutional Networks".
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="growth_rate">How many filters to add each layer.</param>
            /// <param name="bn_size">Multiplicative factor for number of bottleneck layers (i.e. bn_size * k features in the bottleneck layer).</param>
            /// <param name="drop_rate">Dropout rate after each dense layer.</param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last linear layer of the classifier will not be loaded from the weights file.</param>
            /// <param name="device">The device to locate the model on.</param>
            /// <remarks>
            /// Pre-trained weights may be retrieved by using Pytorch and saving the model state-dict
            /// using the exportsd.py script, then loading into the .NET instance:
            ///
            /// from torchvision import models
            /// import exportsd
            ///
            /// model = models.densenet121(pretrained=True)
            /// f = open("model_weights.dat", "wb")
            /// exportsd.save_state_dict(model.state_dict(), f)
            /// f.close()
            ///
            /// See also: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/saveload.md
            ///
            /// In order for the weights to be loaded, the number of classes has to be the same as
            /// in the pre-trained model, which is 1000.
            ///
            /// It is also possible to skip loading the last linear layer and use it for transfer-learning
            /// with a different number of output classes. To do so, pass skipfc=true.
            ///
            /// All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.DenseNet densenet121(
                int num_classes = 1000,
                int growth_rate = 32,
                int bn_size = 4,
                float drop_rate = 0,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return new Modules.DenseNet(growth_rate, new int[] { 6, 12, 24, 16 }, 64, bn_size, drop_rate,
                    num_classes, weights_file, skipfc, device);
            }

            /// <summary>
            /// DenseNet-161 model from "Densely Connected Convolutional Networks".
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="growth_rate">How many filters to add each layer.</param>
            /// <param name="bn_size">Multiplicative factor for number of bottleneck layers.</param>
            /// <param name="drop_rate">Dropout rate after each dense layer.</param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last linear layer of the classifier will not be loaded from the weights file.</param>
            /// <param name="device">The device to locate the model on.</param>
            public static Modules.DenseNet densenet161(
                int num_classes = 1000,
                int growth_rate = 48,
                int bn_size = 4,
                float drop_rate = 0,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return new Modules.DenseNet(growth_rate, new int[] { 6, 12, 36, 24 }, 96, bn_size, drop_rate,
                    num_classes, weights_file, skipfc, device);
            }

            /// <summary>
            /// DenseNet-169 model from "Densely Connected Convolutional Networks".
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="growth_rate">How many filters to add each layer.</param>
            /// <param name="bn_size">Multiplicative factor for number of bottleneck layers.</param>
            /// <param name="drop_rate">Dropout rate after each dense layer.</param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last linear layer of the classifier will not be loaded from the weights file.</param>
            /// <param name="device">The device to locate the model on.</param>
            public static Modules.DenseNet densenet169(
                int num_classes = 1000,
                int growth_rate = 32,
                int bn_size = 4,
                float drop_rate = 0,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return new Modules.DenseNet(growth_rate, new int[] { 6, 12, 32, 32 }, 64, bn_size, drop_rate,
                    num_classes, weights_file, skipfc, device);
            }

            /// <summary>
            /// DenseNet-201 model from "Densely Connected Convolutional Networks".
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="growth_rate">How many filters to add each layer.</param>
            /// <param name="bn_size">Multiplicative factor for number of bottleneck layers.</param>
            /// <param name="drop_rate">Dropout rate after each dense layer.</param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last linear layer of the classifier will not be loaded from the weights file.</param>
            /// <param name="device">The device to locate the model on.</param>
            public static Modules.DenseNet densenet201(
                int num_classes = 1000,
                int growth_rate = 32,
                int bn_size = 4,
                float drop_rate = 0,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return new Modules.DenseNet(growth_rate, new int[] { 6, 12, 48, 32 }, 64, bn_size, drop_rate,
                    num_classes, weights_file, skipfc, device);
            }
        }
    }

    namespace Modules
    {
        // Based on https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
        // License: https://github.com/pytorch/vision/blob/main/LICENSE

        public class DenseNet : Module<Tensor, Tensor>
        {
            /// <summary>
            /// A single dense layer (BN-ReLU-Conv1x1-BN-ReLU-Conv3x3) as described in the paper.
            /// </summary>
            private class DenseLayer : Module<Tensor, Tensor>
            {
                private readonly Module<Tensor, Tensor> norm1;
                private readonly Module<Tensor, Tensor> relu1;
                private readonly Module<Tensor, Tensor> conv1;
                private readonly Module<Tensor, Tensor> norm2;
                private readonly Module<Tensor, Tensor> relu2;
                private readonly Module<Tensor, Tensor> conv2;
                private readonly float drop_rate;

                public DenseLayer(string name, int num_input_features, int growth_rate, int bn_size, float drop_rate)
                    : base(name)
                {
                    norm1 = BatchNorm2d(num_input_features);
                    relu1 = ReLU(inplace: true);
                    conv1 = Conv2d(num_input_features, bn_size * growth_rate, kernel_size: 1, stride: 1, bias: false);
                    norm2 = BatchNorm2d(bn_size * growth_rate);
                    relu2 = ReLU(inplace: true);
                    conv2 = Conv2d(bn_size * growth_rate, growth_rate, kernel_size: 3, stride: 1, padding: 1, bias: false);
                    this.drop_rate = drop_rate;
                    RegisterComponents();
                }

                protected override void Dispose(bool disposing)
                {
                    if (disposing) {
                        norm1.Dispose(); relu1.Dispose(); conv1.Dispose();
                        norm2.Dispose(); relu2.Dispose(); conv2.Dispose();
                    }
                    base.Dispose(disposing);
                }

                public override Tensor forward(Tensor input)
                {
                    var bottleneck_output = conv1.call(relu1.call(norm1.call(input)));
                    var new_features = conv2.call(relu2.call(norm2.call(bottleneck_output)));
                    if (drop_rate > 0 && training)
                        new_features = nn.functional.dropout(new_features, drop_rate, training);
                    return new_features;
                }
            }

            /// <summary>
            /// A dense block consisting of multiple dense layers with progressive feature concatenation.
            /// </summary>
            private class DenseBlock : Module<Tensor, Tensor>
            {
                private readonly Module<Tensor, Tensor>[] denselayers;

                public DenseBlock(string name, int num_layers, int num_input_features, int bn_size, int growth_rate, float drop_rate)
                    : base(name)
                {
                    denselayers = new Module<Tensor, Tensor>[num_layers];
                    for (int i = 0; i < num_layers; i++) {
                        var layer = new DenseLayer($"denselayer{i + 1}",
                            num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate);
                        denselayers[i] = layer;
                        // Use register_module to ensure correct named hierarchy for state_dict compatibility
                        register_module($"denselayer{i + 1}", layer);
                    }
                }

                protected override void Dispose(bool disposing)
                {
                    if (disposing) {
                        foreach (var layer in denselayers)
                            layer.Dispose();
                    }
                    base.Dispose(disposing);
                }

                public override Tensor forward(Tensor init_features)
                {
                    var features = new List<Tensor> { init_features };
                    foreach (var layer in denselayers) {
                        var concat_features = torch.cat(features.ToArray(), 1);
                        var new_features = layer.call(concat_features);
                        features.Add(new_features);
                    }
                    return torch.cat(features.ToArray(), 1);
                }
            }

            /// <summary>
            /// A transition layer (BN-ReLU-Conv1x1-AvgPool) that reduces feature map size.
            /// </summary>
            private class Transition : Module<Tensor, Tensor>
            {
                private readonly Module<Tensor, Tensor> norm;
                private readonly Module<Tensor, Tensor> relu;
                private readonly Module<Tensor, Tensor> conv;
                private readonly Module<Tensor, Tensor> pool;

                public Transition(string name, int num_input_features, int num_output_features) : base(name)
                {
                    norm = BatchNorm2d(num_input_features);
                    relu = ReLU(inplace: true);
                    conv = Conv2d(num_input_features, num_output_features, kernel_size: 1, stride: 1, bias: false);
                    pool = AvgPool2d(kernel_size: 2, stride: 2);
                    RegisterComponents();
                }

                protected override void Dispose(bool disposing)
                {
                    if (disposing) {
                        norm.Dispose(); relu.Dispose(); conv.Dispose(); pool.Dispose();
                    }
                    base.Dispose(disposing);
                }

                public override Tensor forward(Tensor x)
                {
                    return pool.call(conv.call(relu.call(norm.call(x))));
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

            /// <summary>
            /// DenseNet model class.
            /// </summary>
            /// <param name="growth_rate">How many filters to add each layer.</param>
            /// <param name="block_config">Number of layers in each dense block.</param>
            /// <param name="num_init_features">Number of filters in the first convolution layer.</param>
            /// <param name="bn_size">Multiplicative factor for number of bottleneck layers.</param>
            /// <param name="drop_rate">Dropout rate after each dense layer.</param>
            /// <param name="num_classes">Number of output classes.</param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last linear layer will not be loaded from the weights file.</param>
            /// <param name="device">The device to locate the model on.</param>
            public DenseNet(
                int growth_rate = 32,
                int[]? block_config = null,
                int num_init_features = 64,
                int bn_size = 4,
                float drop_rate = 0,
                int num_classes = 1000,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null) : base(nameof(DenseNet))
            {
                if (block_config == null)
                    block_config = new int[] { 6, 12, 24, 16 };

                // Build the features Sequential with named children
                var f = Sequential();
                f.append("conv0", Conv2d(3, num_init_features, kernel_size: 7, stride: 2, padding: 3, bias: false));
                f.append("norm0", BatchNorm2d(num_init_features));
                f.append("relu0", ReLU(inplace: true));
                f.append("pool0", MaxPool2d(kernel_size: 3, stride: 2, padding: 1));

                int num_features = num_init_features;
                for (int i = 0; i < block_config.Length; i++) {
                    var block = new DenseBlock("DenseBlock",
                        block_config[i], num_features, bn_size, growth_rate, drop_rate);
                    f.append($"denseblock{i + 1}", block);
                    num_features = num_features + block_config[i] * growth_rate;
                    if (i != block_config.Length - 1) {
                        var trans = new Transition("Transition",
                            num_features, num_features / 2);
                        f.append($"transition{i + 1}", trans);
                        num_features = num_features / 2;
                    }
                }

                f.append("norm5", BatchNorm2d(num_features));
                features = f;

                classifier = Linear(num_features, num_classes);

                RegisterComponents();

                // Weight initialization
                if (string.IsNullOrEmpty(weights_file)) {
                    foreach (var (_, m) in named_modules()) {
                        if (m is Modules.Conv2d conv) {
                            nn.init.kaiming_normal_(conv.weight);
                        } else if (m is Modules.BatchNorm2d bn) {
                            nn.init.constant_(bn.weight, 1);
                            nn.init.constant_(bn.bias, 0);
                        } else if (m is Modules.Linear linear) {
                            nn.init.constant_(linear.bias, 0);
                        }
                    }
                } else {
                    this.load(weights_file!, skip: skipfc ? new[] { "classifier.weight", "classifier.bias" } : null);
                }

                if (device != null && device.type != DeviceType.CPU)
                    this.to(device);
            }

            public override Tensor forward(Tensor x)
            {
                using (var _ = NewDisposeScope()) {
                    x = features.call(x);
                    x = nn.functional.relu(x);
                    x = nn.functional.adaptive_avg_pool2d(x, new long[] { 1, 1 });
                    x = torch.flatten(x, 1);
                    return classifier.call(x).MoveToOuterDisposeScope();
                }
            }
        }
    }
}
