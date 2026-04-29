// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torchvision.models._utils;
using static TorchSharp.torchvision.ops;
using TorchSharp.Modules;

#nullable enable
namespace TorchSharp
{
    namespace Modules
    {
        public class EfficientNet : nn.Module<Tensor, Tensor>
        {
            internal enum BlockType { MBConv, FusedMBConv }

            /// <summary>
            /// Stores information listed at Tables 1 and 4 of the EfficientNet papers.
            /// </summary>
            internal class _MBConvConfig
            {
                public double expand_ratio;
                public long kernel;
                public long stride;
                public long input_channels;
                public long out_channels;
                public long num_layers;
                public BlockType block_type;

                public _MBConvConfig(
                    double expand_ratio, long kernel, long stride,
                    long input_channels, long out_channels, long num_layers,
                    BlockType block_type)
                {
                    this.expand_ratio = expand_ratio;
                    this.kernel = kernel;
                    this.stride = stride;
                    this.input_channels = input_channels;
                    this.out_channels = out_channels;
                    this.num_layers = num_layers;
                    this.block_type = block_type;
                }

                public static long adjust_channels(long channels, double width_mult, long? min_value = null)
                {
                    return _make_divisible(channels * width_mult, 8, min_value);
                }

                public _MBConvConfig ShallowCopy()
                {
                    return (_MBConvConfig)this.MemberwiseClone();
                }
            }

            /// <summary>
            /// Config for MBConv blocks (EfficientNet B0-B7).
            /// Applies width and depth multipliers for compound scaling.
            /// </summary>
            internal class MBConvConfig : _MBConvConfig
            {
                public MBConvConfig(
                    double expand_ratio, long kernel, long stride,
                    long input_channels, long out_channels, long num_layers,
                    double width_mult = 1.0, double depth_mult = 1.0)
                    : base(expand_ratio, kernel, stride,
                          adjust_channels(input_channels, width_mult),
                          adjust_channels(out_channels, width_mult),
                          adjust_depth(num_layers, depth_mult),
                          BlockType.MBConv)
                {
                }

                public static long adjust_depth(long num_layers, double depth_mult)
                {
                    return (long)Math.Ceiling(num_layers * depth_mult);
                }
            }

            /// <summary>
            /// Config for FusedMBConv blocks (EfficientNet V2).
            /// </summary>
            internal class FusedMBConvConfig : _MBConvConfig
            {
                public FusedMBConvConfig(
                    double expand_ratio, long kernel, long stride,
                    long input_channels, long out_channels, long num_layers)
                    : base(expand_ratio, kernel, stride,
                          input_channels, out_channels, num_layers,
                          BlockType.FusedMBConv)
                {
                }
            }

            /// <summary>
            /// MBConv block: Mobile Inverted Bottleneck Conv with Squeeze-and-Excitation.
            /// </summary>
            private class MBConv : nn.Module<Tensor, Tensor>
            {
                private readonly nn.Module<Tensor, Tensor> block;
                private readonly torchvision.StochasticDepth stochastic_depth;
                private readonly bool use_res_connect;

                protected override void Dispose(bool disposing)
                {
                    if (disposing) {
                        block.Dispose();
                        stochastic_depth.Dispose();
                    }
                    base.Dispose(disposing);
                }

                public MBConv(
                    string name,
                    _MBConvConfig cnf,
                    double stochastic_depth_prob,
                    Func<long, nn.Module<Tensor, Tensor>> norm_layer) : base(name)
                {
                    if (!(1 <= cnf.stride && cnf.stride <= 2))
                        throw new ArgumentException("illegal stride value");

                    use_res_connect = cnf.stride == 1 && cnf.input_channels == cnf.out_channels;

                    var layers = new List<nn.Module<Tensor, Tensor>>();
                    Func<bool, nn.Module<Tensor, Tensor>> activation_layer = (inplace) => nn.SiLU(inplace);

                    // expand
                    var expanded_channels = _MBConvConfig.adjust_channels(cnf.input_channels, cnf.expand_ratio);
                    if (expanded_channels != cnf.input_channels) {
                        layers.Add(Conv2dNormActivation(
                            cnf.input_channels, expanded_channels,
                            kernel_size: 1,
                            norm_layer: norm_layer,
                            activation_layer: activation_layer));
                    }

                    // depthwise
                    layers.Add(Conv2dNormActivation(
                        expanded_channels, expanded_channels,
                        kernel_size: cnf.kernel,
                        stride: cnf.stride,
                        groups: expanded_channels,
                        norm_layer: norm_layer,
                        activation_layer: activation_layer));

                    // squeeze and excitation
                    var squeeze_channels = Math.Max(1, cnf.input_channels / 4);
                    layers.Add(
                        torchvision.ops.SqueezeExcitation(
                            expanded_channels,
                            squeeze_channels,
                            activation: () => nn.SiLU(inplace: true)));

                    // project
                    layers.Add(Conv2dNormActivation(
                        expanded_channels, cnf.out_channels,
                        kernel_size: 1,
                        norm_layer: norm_layer,
                        activation_layer: null));

                    block = nn.Sequential(layers);
                    stochastic_depth = torchvision.ops.StochasticDepth(stochastic_depth_prob, torchvision.StochasticDepth.Mode.Row);

                    RegisterComponents();
                }

                public override Tensor forward(Tensor input)
                {
                    var result = block.call(input);
                    if (use_res_connect) {
                        result = stochastic_depth.call(result);
                        result += input;
                    }
                    return result;
                }
            }

            /// <summary>
            /// FusedMBConv block: Fused Mobile Inverted Bottleneck Conv (no depthwise or SE).
            /// </summary>
            private class FusedMBConv : nn.Module<Tensor, Tensor>
            {
                private readonly nn.Module<Tensor, Tensor> block;
                private readonly torchvision.StochasticDepth stochastic_depth;
                private readonly bool use_res_connect;

                protected override void Dispose(bool disposing)
                {
                    if (disposing) {
                        block.Dispose();
                        stochastic_depth.Dispose();
                    }
                    base.Dispose(disposing);
                }

                public FusedMBConv(
                    string name,
                    _MBConvConfig cnf,
                    double stochastic_depth_prob,
                    Func<long, nn.Module<Tensor, Tensor>> norm_layer) : base(name)
                {
                    if (!(1 <= cnf.stride && cnf.stride <= 2))
                        throw new ArgumentException("illegal stride value");

                    use_res_connect = cnf.stride == 1 && cnf.input_channels == cnf.out_channels;

                    var layers = new List<nn.Module<Tensor, Tensor>>();
                    Func<bool, nn.Module<Tensor, Tensor>> activation_layer = (inplace) => nn.SiLU(inplace);

                    var expanded_channels = _MBConvConfig.adjust_channels(cnf.input_channels, cnf.expand_ratio);
                    if (expanded_channels != cnf.input_channels) {
                        // fused expand
                        layers.Add(Conv2dNormActivation(
                            cnf.input_channels, expanded_channels,
                            kernel_size: cnf.kernel,
                            stride: cnf.stride,
                            norm_layer: norm_layer,
                            activation_layer: activation_layer));

                        // project
                        layers.Add(Conv2dNormActivation(
                            expanded_channels, cnf.out_channels,
                            kernel_size: 1,
                            norm_layer: norm_layer,
                            activation_layer: null));
                    } else {
                        layers.Add(Conv2dNormActivation(
                            cnf.input_channels, cnf.out_channels,
                            kernel_size: cnf.kernel,
                            stride: cnf.stride,
                            norm_layer: norm_layer,
                            activation_layer: activation_layer));
                    }

                    block = nn.Sequential(layers);
                    stochastic_depth = torchvision.ops.StochasticDepth(stochastic_depth_prob, torchvision.StochasticDepth.Mode.Row);

                    RegisterComponents();
                }

                public override Tensor forward(Tensor input)
                {
                    var result = block.call(input);
                    if (use_res_connect) {
                        result = stochastic_depth.call(result);
                        result += input;
                    }
                    return result;
                }
            }

            private readonly nn.Module<Tensor, Tensor> features;
            private readonly nn.Module<Tensor, Tensor> avgpool;
            private readonly nn.Module<Tensor, Tensor> classifier;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    features.Dispose();
                    avgpool.Dispose();
                    classifier.Dispose();
                }
                base.Dispose(disposing);
            }

            /// <summary>
            /// EfficientNet V1 and V2 main class
            /// </summary>
            /// <param name="name"></param>
            /// <param name="inverted_residual_setting">Network structure</param>
            /// <param name="dropout">The dropout probability</param>
            /// <param name="stochastic_depth_prob">The stochastic depth probability</param>
            /// <param name="num_classes">Number of classes</param>
            /// <param name="norm_layer">Module specifying the normalization layer to use</param>
            /// <param name="last_channel">The number of channels on the penultimate layer</param>
            internal EfficientNet(
                string name,
                _MBConvConfig[] inverted_residual_setting,
                double dropout,
                double stochastic_depth_prob = 0.2,
                long num_classes = 1000,
                Func<long, nn.Module<Tensor, Tensor>>? norm_layer = null,
                long? last_channel = null) : base(name)
            {
                if (inverted_residual_setting == null || inverted_residual_setting.Length == 0)
                    throw new ArgumentException("The inverted_residual_setting should not be empty");

                if (norm_layer == null)
                    norm_layer = (features) => nn.BatchNorm2d(features);

                var layers = new List<nn.Module<Tensor, Tensor>>();

                // building first layer
                var firstconv_output_channels = inverted_residual_setting[0].input_channels;
                layers.Add(Conv2dNormActivation(
                    3, firstconv_output_channels,
                    kernel_size: 3, stride: 2,
                    norm_layer: norm_layer,
                    activation_layer: (inplace) => nn.SiLU(inplace)));

                // building inverted residual blocks
                long total_stage_blocks = 0;
                foreach (var cnf in inverted_residual_setting)
                    total_stage_blocks += cnf.num_layers;

                long stage_block_id = 0;
                foreach (var cnf in inverted_residual_setting) {
                    var stage = new List<nn.Module<Tensor, Tensor>>();
                    for (int i = 0; i < cnf.num_layers; i++) {
                        var block_cnf = cnf.ShallowCopy();

                        // overwrite info if not the first conv in the stage
                        if (stage.Count > 0) {
                            block_cnf.input_channels = block_cnf.out_channels;
                            block_cnf.stride = 1;
                        }

                        // adjust stochastic depth probability based on the depth of the stage block
                        var sd_prob = stochastic_depth_prob * (double)stage_block_id / total_stage_blocks;

                        if (block_cnf.block_type == BlockType.FusedMBConv) {
                            stage.Add(new FusedMBConv("FusedMBConv", block_cnf, sd_prob, norm_layer));
                        } else {
                            stage.Add(new MBConv("MBConv", block_cnf, sd_prob, norm_layer));
                        }
                        stage_block_id++;
                    }
                    layers.Add(nn.Sequential(stage));
                }

                // building last several layers
                var lastconv_input_channels = inverted_residual_setting[inverted_residual_setting.Length - 1].out_channels;
                var lastconv_output_channels = last_channel.HasValue ? last_channel.Value : 4 * lastconv_input_channels;
                layers.Add(Conv2dNormActivation(
                    lastconv_input_channels, lastconv_output_channels,
                    kernel_size: 1,
                    norm_layer: norm_layer,
                    activation_layer: (inplace) => nn.SiLU(inplace)));

                features = nn.Sequential(layers);
                avgpool = nn.AdaptiveAvgPool2d(1);
                classifier = nn.Sequential(
                    nn.Dropout(p: dropout, inplace: true),
                    nn.Linear(lastconv_output_channels, num_classes));

                RegisterComponents();

                foreach (var (_, m) in this.named_modules()) {
                    if (m is Modules.Conv2d) {
                        var conv = (Modules.Conv2d)m;
                        nn.init.kaiming_normal_(conv.weight, mode: nn.init.FanInOut.FanOut);
                        if (conv.bias is not null) {
                            nn.init.zeros_(conv.bias);
                        }
                    } else if (m is Modules.BatchNorm2d) {
                        var norm = (Modules.BatchNorm2d)m;
                        nn.init.ones_(norm.weight);
                        nn.init.zeros_(norm.bias);
                    } else if (m is Modules.GroupNorm) {
                        var norm = (Modules.GroupNorm)m;
                        nn.init.ones_(norm.weight);
                        nn.init.zeros_(norm.bias);
                    } else if (m is Modules.Linear) {
                        var linear = (Modules.Linear)m;
                        var init_range = 1.0 / Math.Sqrt(linear.weight.shape[0]);
                        nn.init.uniform_(linear.weight, -init_range, init_range);
                        nn.init.zeros_(linear.bias);
                    }
                }
            }

            public override Tensor forward(Tensor x)
            {
                using (var _ = NewDisposeScope()) {
                    x = features.call(x);
                    x = avgpool.call(x);
                    x = torch.flatten(x, 1);
                    x = classifier.call(x);
                    return x.MoveToOuterDisposeScope();
                }
            }
        }
    }

    public static partial class torchvision
    {
        public static partial class models
        {
            private static (EfficientNet._MBConvConfig[], long?) _efficientnet_conf(string arch, double width_mult = 1.0, double depth_mult = 1.0)
            {
                EfficientNet._MBConvConfig[] inverted_residual_setting;
                long? last_channel;

                if (arch.StartsWith("efficientnet_b")) {
                    EfficientNet._MBConvConfig bneck_conf(
                        double expand_ratio, long kernel, long stride,
                        long input_channels, long out_channels, long num_layers) =>
                        new EfficientNet.MBConvConfig(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, width_mult, depth_mult);

                    inverted_residual_setting = new EfficientNet._MBConvConfig[] {
                        bneck_conf(1, 3, 1, 32, 16, 1),
                        bneck_conf(6, 3, 2, 16, 24, 2),
                        bneck_conf(6, 5, 2, 24, 40, 2),
                        bneck_conf(6, 3, 2, 40, 80, 3),
                        bneck_conf(6, 5, 1, 80, 112, 3),
                        bneck_conf(6, 5, 2, 112, 192, 4),
                        bneck_conf(6, 3, 1, 192, 320, 1),
                    };
                    last_channel = null;
                } else if (arch.StartsWith("efficientnet_v2_s")) {
                    inverted_residual_setting = new EfficientNet._MBConvConfig[] {
                        new EfficientNet.FusedMBConvConfig(1, 3, 1, 24, 24, 2),
                        new EfficientNet.FusedMBConvConfig(4, 3, 2, 24, 48, 4),
                        new EfficientNet.FusedMBConvConfig(4, 3, 2, 48, 64, 4),
                        new EfficientNet.MBConvConfig(4, 3, 2, 64, 128, 6),
                        new EfficientNet.MBConvConfig(6, 3, 1, 128, 160, 9),
                        new EfficientNet.MBConvConfig(6, 3, 2, 160, 256, 15),
                    };
                    last_channel = 1280;
                } else if (arch.StartsWith("efficientnet_v2_m")) {
                    inverted_residual_setting = new EfficientNet._MBConvConfig[] {
                        new EfficientNet.FusedMBConvConfig(1, 3, 1, 24, 24, 3),
                        new EfficientNet.FusedMBConvConfig(4, 3, 2, 24, 48, 5),
                        new EfficientNet.FusedMBConvConfig(4, 3, 2, 48, 80, 5),
                        new EfficientNet.MBConvConfig(4, 3, 2, 80, 160, 7),
                        new EfficientNet.MBConvConfig(6, 3, 1, 160, 176, 14),
                        new EfficientNet.MBConvConfig(6, 3, 2, 176, 304, 18),
                        new EfficientNet.MBConvConfig(6, 3, 1, 304, 512, 5),
                    };
                    last_channel = 1280;
                } else if (arch.StartsWith("efficientnet_v2_l")) {
                    inverted_residual_setting = new EfficientNet._MBConvConfig[] {
                        new EfficientNet.FusedMBConvConfig(1, 3, 1, 32, 32, 4),
                        new EfficientNet.FusedMBConvConfig(4, 3, 2, 32, 64, 7),
                        new EfficientNet.FusedMBConvConfig(4, 3, 2, 64, 96, 7),
                        new EfficientNet.MBConvConfig(4, 3, 2, 96, 192, 10),
                        new EfficientNet.MBConvConfig(6, 3, 1, 192, 224, 19),
                        new EfficientNet.MBConvConfig(6, 3, 2, 224, 384, 25),
                        new EfficientNet.MBConvConfig(6, 3, 1, 384, 640, 7),
                    };
                    last_channel = 1280;
                } else {
                    throw new ArgumentException($"Unsupported model type {arch}");
                }

                return (inverted_residual_setting, last_channel);
            }

            private static Modules.EfficientNet _efficientnet(
                EfficientNet._MBConvConfig[] inverted_residual_setting,
                double dropout,
                long? last_channel,
                long num_classes = 1000,
                Func<long, nn.Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                var model = new EfficientNet("EfficientNet", inverted_residual_setting, dropout, num_classes: num_classes, norm_layer: norm_layer, last_channel: last_channel);

                if (!string.IsNullOrEmpty(weights_file)) {
                    model.load(weights_file!, skip: skipfc ? new[] { "classifier.1.weight", "classifier.1.bias" } : null);
                }

                if (device != null && device.type != DeviceType.CPU)
                    model.to(device);

                return model;
            }

            /// <summary>
            /// EfficientNet B0 model architecture from the
            /// <a href="https://arxiv.org/abs/1905.11946">EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a> paper.
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.efficientnet_b0(weights='DEFAULT')
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
            public static Modules.EfficientNet efficientnet_b0(int num_classes = 1000, float dropout = 0.2f, string? weights_file = null, bool skipfc = true, Device? device = null)
            {
                var (inverted_residual_setting, last_channel) = _efficientnet_conf("efficientnet_b0", width_mult: 1.0, depth_mult: 1.0);
                return _efficientnet(inverted_residual_setting, dropout, last_channel, num_classes, weights_file: weights_file, skipfc: skipfc, device: device);
            }

            /// <summary>
            /// EfficientNet B1 model architecture from the
            /// <a href="https://arxiv.org/abs/1905.11946">EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a> paper.
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.efficientnet_b1(weights='DEFAULT')
            /// f = open("model_weights.dat", "wb")
            /// exportsd.save_state_dict(model.state_dict(), f)
            /// f.close()
            ///
            /// See also: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/saveload.md
            /// </remarks>
            public static Modules.EfficientNet efficientnet_b1(int num_classes = 1000, float dropout = 0.2f, string? weights_file = null, bool skipfc = true, Device? device = null)
            {
                var (inverted_residual_setting, last_channel) = _efficientnet_conf("efficientnet_b1", width_mult: 1.0, depth_mult: 1.1);
                return _efficientnet(inverted_residual_setting, dropout, last_channel, num_classes, weights_file: weights_file, skipfc: skipfc, device: device);
            }

            /// <summary>
            /// EfficientNet B2 model architecture from the
            /// <a href="https://arxiv.org/abs/1905.11946">EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a> paper.
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.efficientnet_b2(weights='DEFAULT')
            /// f = open("model_weights.dat", "wb")
            /// exportsd.save_state_dict(model.state_dict(), f)
            /// f.close()
            ///
            /// See also: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/saveload.md
            /// </remarks>
            public static Modules.EfficientNet efficientnet_b2(int num_classes = 1000, float dropout = 0.3f, string? weights_file = null, bool skipfc = true, Device? device = null)
            {
                var (inverted_residual_setting, last_channel) = _efficientnet_conf("efficientnet_b2", width_mult: 1.1, depth_mult: 1.2);
                return _efficientnet(inverted_residual_setting, dropout, last_channel, num_classes, weights_file: weights_file, skipfc: skipfc, device: device);
            }

            /// <summary>
            /// EfficientNet B3 model architecture from the
            /// <a href="https://arxiv.org/abs/1905.11946">EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a> paper.
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.efficientnet_b3(weights='DEFAULT')
            /// f = open("model_weights.dat", "wb")
            /// exportsd.save_state_dict(model.state_dict(), f)
            /// f.close()
            ///
            /// See also: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/saveload.md
            /// </remarks>
            public static Modules.EfficientNet efficientnet_b3(int num_classes = 1000, float dropout = 0.3f, string? weights_file = null, bool skipfc = true, Device? device = null)
            {
                var (inverted_residual_setting, last_channel) = _efficientnet_conf("efficientnet_b3", width_mult: 1.2, depth_mult: 1.4);
                return _efficientnet(inverted_residual_setting, dropout, last_channel, num_classes, weights_file: weights_file, skipfc: skipfc, device: device);
            }

            /// <summary>
            /// EfficientNet B4 model architecture from the
            /// <a href="https://arxiv.org/abs/1905.11946">EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a> paper.
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.efficientnet_b4(weights='DEFAULT')
            /// f = open("model_weights.dat", "wb")
            /// exportsd.save_state_dict(model.state_dict(), f)
            /// f.close()
            ///
            /// See also: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/saveload.md
            /// </remarks>
            public static Modules.EfficientNet efficientnet_b4(int num_classes = 1000, float dropout = 0.4f, string? weights_file = null, bool skipfc = true, Device? device = null)
            {
                var (inverted_residual_setting, last_channel) = _efficientnet_conf("efficientnet_b4", width_mult: 1.4, depth_mult: 1.8);
                return _efficientnet(inverted_residual_setting, dropout, last_channel, num_classes, weights_file: weights_file, skipfc: skipfc, device: device);
            }

            /// <summary>
            /// EfficientNet B5 model architecture from the
            /// <a href="https://arxiv.org/abs/1905.11946">EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a> paper.
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.efficientnet_b5(weights='DEFAULT')
            /// f = open("model_weights.dat", "wb")
            /// exportsd.save_state_dict(model.state_dict(), f)
            /// f.close()
            ///
            /// See also: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/saveload.md
            /// </remarks>
            public static Modules.EfficientNet efficientnet_b5(int num_classes = 1000, float dropout = 0.4f, string? weights_file = null, bool skipfc = true, Device? device = null)
            {
                var (inverted_residual_setting, last_channel) = _efficientnet_conf("efficientnet_b5", width_mult: 1.6, depth_mult: 2.2);
                Func<long, nn.Module<Tensor, Tensor>> norm_layer = (features) => nn.BatchNorm2d(features, eps: 0.001, momentum: 0.01);
                return _efficientnet(inverted_residual_setting, dropout, last_channel, num_classes, norm_layer: norm_layer, weights_file: weights_file, skipfc: skipfc, device: device);
            }

            /// <summary>
            /// EfficientNet B6 model architecture from the
            /// <a href="https://arxiv.org/abs/1905.11946">EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a> paper.
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.efficientnet_b6(weights='DEFAULT')
            /// f = open("model_weights.dat", "wb")
            /// exportsd.save_state_dict(model.state_dict(), f)
            /// f.close()
            ///
            /// See also: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/saveload.md
            /// </remarks>
            public static Modules.EfficientNet efficientnet_b6(int num_classes = 1000, float dropout = 0.5f, string? weights_file = null, bool skipfc = true, Device? device = null)
            {
                var (inverted_residual_setting, last_channel) = _efficientnet_conf("efficientnet_b6", width_mult: 1.8, depth_mult: 2.6);
                Func<long, nn.Module<Tensor, Tensor>> norm_layer = (features) => nn.BatchNorm2d(features, eps: 0.001, momentum: 0.01);
                return _efficientnet(inverted_residual_setting, dropout, last_channel, num_classes, norm_layer: norm_layer, weights_file: weights_file, skipfc: skipfc, device: device);
            }

            /// <summary>
            /// EfficientNet B7 model architecture from the
            /// <a href="https://arxiv.org/abs/1905.11946">EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a> paper.
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.efficientnet_b7(weights='DEFAULT')
            /// f = open("model_weights.dat", "wb")
            /// exportsd.save_state_dict(model.state_dict(), f)
            /// f.close()
            ///
            /// See also: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/saveload.md
            /// </remarks>
            public static Modules.EfficientNet efficientnet_b7(int num_classes = 1000, float dropout = 0.5f, string? weights_file = null, bool skipfc = true, Device? device = null)
            {
                var (inverted_residual_setting, last_channel) = _efficientnet_conf("efficientnet_b7", width_mult: 2.0, depth_mult: 3.1);
                Func<long, nn.Module<Tensor, Tensor>> norm_layer = (features) => nn.BatchNorm2d(features, eps: 0.001, momentum: 0.01);
                return _efficientnet(inverted_residual_setting, dropout, last_channel, num_classes, norm_layer: norm_layer, weights_file: weights_file, skipfc: skipfc, device: device);
            }

            /// <summary>
            /// Constructs an EfficientNetV2-S architecture from
            /// <a href="https://arxiv.org/abs/2104.00298">EfficientNetV2: Smaller Models and Faster Training</a>.
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.efficientnet_v2_s(weights='DEFAULT')
            /// f = open("model_weights.dat", "wb")
            /// exportsd.save_state_dict(model.state_dict(), f)
            /// f.close()
            ///
            /// See also: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/saveload.md
            /// </remarks>
            public static Modules.EfficientNet efficientnet_v2_s(int num_classes = 1000, float dropout = 0.2f, string? weights_file = null, bool skipfc = true, Device? device = null)
            {
                var (inverted_residual_setting, last_channel) = _efficientnet_conf("efficientnet_v2_s");
                Func<long, nn.Module<Tensor, Tensor>> norm_layer = (features) => nn.BatchNorm2d(features, eps: 0.001);
                return _efficientnet(inverted_residual_setting, dropout, last_channel, num_classes, norm_layer: norm_layer, weights_file: weights_file, skipfc: skipfc, device: device);
            }

            /// <summary>
            /// Constructs an EfficientNetV2-M architecture from
            /// <a href="https://arxiv.org/abs/2104.00298">EfficientNetV2: Smaller Models and Faster Training</a>.
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.efficientnet_v2_m(weights='DEFAULT')
            /// f = open("model_weights.dat", "wb")
            /// exportsd.save_state_dict(model.state_dict(), f)
            /// f.close()
            ///
            /// See also: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/saveload.md
            /// </remarks>
            public static Modules.EfficientNet efficientnet_v2_m(int num_classes = 1000, float dropout = 0.3f, string? weights_file = null, bool skipfc = true, Device? device = null)
            {
                var (inverted_residual_setting, last_channel) = _efficientnet_conf("efficientnet_v2_m");
                Func<long, nn.Module<Tensor, Tensor>> norm_layer = (features) => nn.BatchNorm2d(features, eps: 0.001);
                return _efficientnet(inverted_residual_setting, dropout, last_channel, num_classes, norm_layer: norm_layer, weights_file: weights_file, skipfc: skipfc, device: device);
            }

            /// <summary>
            /// Constructs an EfficientNetV2-L architecture from
            /// <a href="https://arxiv.org/abs/2104.00298">EfficientNetV2: Smaller Models and Faster Training</a>.
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.efficientnet_v2_l(weights='DEFAULT')
            /// f = open("model_weights.dat", "wb")
            /// exportsd.save_state_dict(model.state_dict(), f)
            /// f.close()
            ///
            /// See also: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/saveload.md
            /// </remarks>
            public static Modules.EfficientNet efficientnet_v2_l(int num_classes = 1000, float dropout = 0.4f, string? weights_file = null, bool skipfc = true, Device? device = null)
            {
                var (inverted_residual_setting, last_channel) = _efficientnet_conf("efficientnet_v2_l");
                Func<long, nn.Module<Tensor, Tensor>> norm_layer = (features) => nn.BatchNorm2d(features, eps: 0.001);
                return _efficientnet(inverted_residual_setting, dropout, last_channel, num_classes, norm_layer: norm_layer, weights_file: weights_file, skipfc: skipfc, device: device);
            }
        }
    }
}
