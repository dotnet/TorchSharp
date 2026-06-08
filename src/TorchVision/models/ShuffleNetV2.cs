// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
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
            /// ShuffleNet V2 with 0.5x output channels, as described in
            /// "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design".
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
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
            /// model = models.shufflenet_v2_x0_5(pretrained=True)
            /// f = open("model_weights.dat", "wb")
            /// exportsd.save_state_dict(model.state_dict(), f)
            /// f.close()
            ///
            /// See also: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/saveload.md
            ///
            /// All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.ShuffleNetV2 shufflenet_v2_x0_5(
                int num_classes = 1000,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return new Modules.ShuffleNetV2(
                    new int[] { 4, 8, 4 },
                    new int[] { 24, 48, 96, 192, 1024 },
                    num_classes, weights_file, skipfc, device);
            }

            /// <summary>
            /// ShuffleNet V2 with 1.0x output channels, as described in
            /// "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design".
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last linear layer of the classifier will not be loaded from the weights file.</param>
            /// <param name="device">The device to locate the model on.</param>
            public static Modules.ShuffleNetV2 shufflenet_v2_x1_0(
                int num_classes = 1000,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return new Modules.ShuffleNetV2(
                    new int[] { 4, 8, 4 },
                    new int[] { 24, 116, 232, 464, 1024 },
                    num_classes, weights_file, skipfc, device);
            }

            /// <summary>
            /// ShuffleNet V2 with 1.5x output channels, as described in
            /// "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design".
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last linear layer of the classifier will not be loaded from the weights file.</param>
            /// <param name="device">The device to locate the model on.</param>
            public static Modules.ShuffleNetV2 shufflenet_v2_x1_5(
                int num_classes = 1000,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return new Modules.ShuffleNetV2(
                    new int[] { 4, 8, 4 },
                    new int[] { 24, 176, 352, 704, 1024 },
                    num_classes, weights_file, skipfc, device);
            }

            /// <summary>
            /// ShuffleNet V2 with 2.0x output channels, as described in
            /// "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design".
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last linear layer of the classifier will not be loaded from the weights file.</param>
            /// <param name="device">The device to locate the model on.</param>
            public static Modules.ShuffleNetV2 shufflenet_v2_x2_0(
                int num_classes = 1000,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return new Modules.ShuffleNetV2(
                    new int[] { 4, 8, 4 },
                    new int[] { 24, 244, 488, 976, 2048 },
                    num_classes, weights_file, skipfc, device);
            }
        }
    }

    namespace Modules
    {
        // Based on https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
        // License: https://github.com/pytorch/vision/blob/main/LICENSE

        public class ShuffleNetV2 : Module<Tensor, Tensor>
        {
            private static Tensor channel_shuffle(Tensor x, int groups)
            {
                var batchsize = x.shape[0];
                var num_channels = x.shape[1];
                var height = x.shape[2];
                var width = x.shape[3];
                var channels_per_group = num_channels / groups;

                x = x.view(batchsize, groups, channels_per_group, height, width);
                x = x.transpose(1, 2).contiguous();
                x = x.view(batchsize, num_channels, height, width);
                return x;
            }

            private static Module<Tensor, Tensor> depthwise_conv(
                long i, long o, long kernel_size, long stride = 1, long padding = 0, bool bias = false)
            {
                return Conv2d(i, o, kernel_size: kernel_size, stride: stride, padding: padding, bias: bias, groups: i);
            }

            private class InvertedResidual : Module<Tensor, Tensor>
            {
                private readonly Module<Tensor, Tensor> branch1;
                private readonly Module<Tensor, Tensor> branch2;
                private readonly int _stride;

                public InvertedResidual(string name, long inp, long oup, int stride) : base(name)
                {
                    if (stride < 1 || stride > 3)
                        throw new ArgumentException("illegal stride value", nameof(stride));

                    _stride = stride;
                    var branch_features = oup / 2;

                    if (stride > 1) {
                        branch1 = Sequential(
                            depthwise_conv(inp, inp, kernel_size: 3, stride: stride, padding: 1),
                            BatchNorm2d(inp),
                            Conv2d(inp, branch_features, kernel_size: 1, stride: 1, padding: 0L, bias: false),
                            BatchNorm2d(branch_features),
                            ReLU(inplace: true)
                        );
                    } else {
                        branch1 = Sequential();
                    }

                    branch2 = Sequential(
                        Conv2d(stride > 1 ? inp : branch_features, branch_features, kernel_size: 1, stride: 1, padding: 0L, bias: false),
                        BatchNorm2d(branch_features),
                        ReLU(inplace: true),
                        depthwise_conv(branch_features, branch_features, kernel_size: 3, stride: stride, padding: 1),
                        BatchNorm2d(branch_features),
                        Conv2d(branch_features, branch_features, kernel_size: 1, stride: 1, padding: 0L, bias: false),
                        BatchNorm2d(branch_features),
                        ReLU(inplace: true)
                    );

                    RegisterComponents();
                }

                protected override void Dispose(bool disposing)
                {
                    if (disposing) {
                        branch1.Dispose();
                        branch2.Dispose();
                    }
                    base.Dispose(disposing);
                }

                public override Tensor forward(Tensor x)
                {
                    Tensor @out;
                    if (_stride == 1) {
                        var chunks = x.chunk(2, dim: 1);
                        @out = torch.cat(new[] { chunks[0], branch2.call(chunks[1]) }, 1);
                    } else {
                        @out = torch.cat(new[] { branch1.call(x), branch2.call(x) }, 1);
                    }
                    @out = channel_shuffle(@out, 2);
                    return @out;
                }
            }

            private readonly Module<Tensor, Tensor> conv1;
            private readonly Module<Tensor, Tensor> maxpool;
            private readonly Module<Tensor, Tensor> stage2;
            private readonly Module<Tensor, Tensor> stage3;
            private readonly Module<Tensor, Tensor> stage4;
            private readonly Module<Tensor, Tensor> conv5;
            private readonly Module<Tensor, Tensor> fc;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    conv1.Dispose(); maxpool.Dispose();
                    stage2.Dispose(); stage3.Dispose(); stage4.Dispose();
                    conv5.Dispose(); fc.Dispose();
                }
                base.Dispose(disposing);
            }

            private static Module<Tensor, Tensor> MakeStage(long input_channels, long output_channels, int repeats)
            {
                var modules = new List<Module<Tensor, Tensor>>();
                modules.Add(new InvertedResidual("InvertedResidual", input_channels, output_channels, 2));
                for (int i = 0; i < repeats - 1; i++) {
                    modules.Add(new InvertedResidual("InvertedResidual", output_channels, output_channels, 1));
                }
                return Sequential(modules.ToArray());
            }

            /// <summary>
            /// ShuffleNet V2 main class.
            /// </summary>
            /// <param name="stages_repeats">Number of repeated blocks in each stage.</param>
            /// <param name="stages_out_channels">Output channels for each stage.</param>
            /// <param name="num_classes">Number of output classes.</param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last linear layer will not be loaded from the weights file.</param>
            /// <param name="device">The device to locate the model on.</param>
            public ShuffleNetV2(
                int[] stages_repeats,
                int[] stages_out_channels,
                int num_classes = 1000,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null) : base(nameof(ShuffleNetV2))
            {
                if (stages_repeats.Length != 3)
                    throw new ArgumentException("expected stages_repeats to have 3 elements");
                if (stages_out_channels.Length != 5)
                    throw new ArgumentException("expected stages_out_channels to have 5 elements");

                long input_channels = 3;
                long output_channels = stages_out_channels[0];

                conv1 = Sequential(
                    Conv2d(input_channels, output_channels, kernel_size: 3, stride: 2, padding: 1, bias: false),
                    BatchNorm2d(output_channels),
                    ReLU(inplace: true)
                );
                input_channels = output_channels;

                maxpool = MaxPool2d(kernel_size: 3, stride: 2, padding: 1);

                stage2 = MakeStage(input_channels, stages_out_channels[1], stages_repeats[0]);
                stage3 = MakeStage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1]);
                stage4 = MakeStage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2]);

                output_channels = stages_out_channels[4];
                conv5 = Sequential(
                    Conv2d(stages_out_channels[3], output_channels, kernel_size: 1, stride: 1, padding: 0L, bias: false),
                    BatchNorm2d(output_channels),
                    ReLU(inplace: true)
                );

                fc = Linear(output_channels, num_classes);

                RegisterComponents();

                if (!string.IsNullOrEmpty(weights_file)) {
                    this.load(weights_file!, skip: skipfc ? new[] { "fc.weight", "fc.bias" } : null);
                }

                if (device != null && device.type != DeviceType.CPU)
                    this.to(device);
            }

            public override Tensor forward(Tensor x)
            {
                using (var _ = NewDisposeScope()) {
                    x = conv1.call(x);
                    x = maxpool.call(x);
                    x = stage2.call(x);
                    x = stage3.call(x);
                    x = stage4.call(x);
                    x = conv5.call(x);
                    x = x.mean(new long[] { 2, 3 }); // global pool
                    x = fc.call(x);
                    return x.MoveToOuterDisposeScope();
                }
            }
        }
    }
}
