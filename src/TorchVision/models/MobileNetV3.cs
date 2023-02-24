// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/3c9ae0ac5dda3d881a6ce5004ce5756ae7de7bc4/torchvision/models/mobilenetv3.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System.Collections.Generic;
using System;
using static TorchSharp.torch;
using static TorchSharp.torchvision.models._utils;
using static TorchSharp.torchvision.ops;
using TorchSharp.Modules;

#nullable enable
namespace TorchSharp
{
    namespace Modules
    {
        public class MobileNetV3 : nn.Module<Tensor, Tensor>
        {
            /// <summary>
            /// Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
            /// </summary>
            internal class InvertedResidualConfig
            {
                public readonly long dilation;
                public readonly long expanded_channels;
                public readonly long input_channels;
                public readonly long kernel;
                public readonly long out_channels;
                public readonly long stride;
                public readonly bool use_hs;
                public readonly bool use_se;

                public InvertedResidualConfig(
                    long input_channels,
                    long kernel,
                    long expanded_channels,
                    long out_channels,
                    bool use_se,
                    string activation,
                    long stride,
                    long dilation,
                    double width_mult)
                {
                    this.input_channels = adjust_channels(input_channels, width_mult);
                    this.kernel = kernel;
                    this.expanded_channels = adjust_channels(expanded_channels, width_mult);
                    this.out_channels = adjust_channels(out_channels, width_mult);
                    this.use_se = use_se;
                    this.use_hs = activation == "HS";
                    this.stride = stride;
                    this.dilation = dilation;
                }

                internal static long adjust_channels(long channels, double width_mult)
                {
                    return _make_divisible(channels * width_mult, 8);
                }
            }

            /// <summary>
            /// Implemented as described at section 5 of MobileNetV3 paper
            /// </summary>
            private class InvertedResidual : nn.Module<Tensor, Tensor>
            {
                private readonly bool _is_cn;
                private readonly nn.Module<Tensor, Tensor> block;
                private readonly long out_channels;
                private readonly bool use_res_connect;

                public InvertedResidual(
                    string name,
                    InvertedResidualConfig cnf,
                    Func<long, nn.Module<Tensor, Tensor>> norm_layer,
                    Func<long, long, nn.Module<Tensor, Tensor>>? se_layer = null) : base(name)
                {
                    if (!(1 <= cnf.stride && cnf.stride <= 2)) {
                        throw new ArgumentException("illegal stride value");
                    }

                    this.use_res_connect = cnf.stride == 1 && cnf.input_channels == cnf.out_channels;

                    var layers = new List<nn.Module<Tensor, Tensor>>();
                    Func<bool, nn.Module<Tensor, Tensor>> activation_layer = (
                        cnf.use_hs ? (inplace) => nn.Hardswish(inplace) : (inplace) => nn.ReLU(inplace));

                    // expand
                    if (cnf.expanded_channels != cnf.input_channels) {
                        layers.Add(Conv2dNormActivation(
                            cnf.input_channels,
                            cnf.expanded_channels,
                            kernel_size: 1,
                            norm_layer: norm_layer,
                            activation_layer: activation_layer));
                    }

                    // depthwise
                    var stride = cnf.dilation > 1 ? 1 : cnf.stride;
                    layers.Add(Conv2dNormActivation(
                        cnf.expanded_channels,
                        cnf.expanded_channels,
                        kernel_size: cnf.kernel,
                        stride: stride,
                        dilation: cnf.dilation,
                        groups: cnf.expanded_channels,
                        norm_layer: norm_layer,
                        activation_layer: activation_layer));
                    if (cnf.use_se) {
                        var squeeze_channels = _make_divisible(cnf.expanded_channels / 4, 8);
                        if (se_layer != null) {
                            layers.Add(se_layer(cnf.expanded_channels, squeeze_channels));
                        } else {
                            layers.Add(
                                new SqueezeExcitation(
                                    "SqueezeExcitation",
                                    cnf.expanded_channels,
                                    squeeze_channels,
                                    activation: () => nn.ReLU6(),
                                    scale_activation: () => nn.Hardsigmoid()));
                        }
                    }

                    // project
                    layers.Add(
                        Conv2dNormActivation(
                            cnf.expanded_channels, cnf.out_channels, kernel_size: 1, norm_layer: norm_layer, activation_layer: null));

                    this.block = nn.Sequential(layers);
                    this.out_channels = cnf.out_channels;
                    this._is_cn = cnf.stride > 1;

                    RegisterComponents();
                }

                public override Tensor forward(Tensor input)
                {
                    var result = this.block.call(input);
                    if (this.use_res_connect) {
                        result += input;
                    }
                    return result;
                }
            }

            private readonly nn.Module<Tensor, Tensor> avgpool;
            private readonly nn.Module<Tensor, Tensor> classifier;
            private readonly nn.Module<Tensor, Tensor> features;

            /// <summary>
            /// MobileNet V3 main class
            /// </summary>
            /// <param name="name"></param>
            /// <param name="inverted_residual_setting">Network structure</param>
            /// <param name="last_channel">The number of channels on the penultimate layer</param>
            /// <param name="num_classes">Number of classes</param>
            /// <param name="block">Module specifying inverted residual building block for mobilenet</param>
            /// <param name="norm_layer">Module specifying the normalization layer to use</param>
            /// <param name="dropout">The droupout probability</param>
            /// <exception cref="ArgumentException"></exception>
            internal MobileNetV3(
                string name,
                InvertedResidualConfig[] inverted_residual_setting,
                long last_channel,
                long num_classes = 1000,
                Func<InvertedResidualConfig, Func<long, nn.Module<Tensor, Tensor>>, nn.Module<Tensor, Tensor>>? block = null,
                Func<long, nn.Module<Tensor, Tensor>>? norm_layer = null,
                double dropout = 0.2) : base(name)
            {
                if (inverted_residual_setting == null || inverted_residual_setting.Length == 0) {
                    throw new ArgumentException("The inverted_residual_setting should not be empty");
                }

                if (block == null) {
                    block = (cnf, norm_layer) => new InvertedResidual("InvertedResidual", cnf, norm_layer);
                }

                if (norm_layer == null) {
                    norm_layer = (features) => nn.BatchNorm2d(features, eps: 0.001, momentum: 0.01);
                }

                var layers = new List<nn.Module<Tensor, Tensor>>();

                // building first layer
                var firstconv_output_channels = inverted_residual_setting[0].input_channels;
                layers.Add(
                    Conv2dNormActivation(
                        3,
                        firstconv_output_channels,
                        kernel_size: 3,
                        stride: 2,
                        norm_layer: norm_layer,
                        activation_layer: (inplace) => nn.Hardswish(inplace)));

                // building inverted residual blocks
                foreach (var cnf in inverted_residual_setting) {
                    layers.Add(block(cnf, norm_layer));
                }

                // building last several layers
                var lastconv_input_channels = inverted_residual_setting[inverted_residual_setting.Length - 1].out_channels;
                var lastconv_output_channels = 6 * lastconv_input_channels;
                layers.Add(
                    Conv2dNormActivation(
                        lastconv_input_channels,
                        lastconv_output_channels,
                        kernel_size: 1,
                        norm_layer: norm_layer,
                        activation_layer: (inplace) => nn.Hardswish(inplace)));

                this.features = nn.Sequential(layers);
                this.avgpool = nn.AdaptiveAvgPool2d(1);
                this.classifier = nn.Sequential(
                    nn.Linear(lastconv_output_channels, last_channel),
                    nn.Hardswish(inplace: true),
                    nn.Dropout(p: dropout, inplace: true),
                    nn.Linear(last_channel, num_classes));

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
                        nn.init.normal_(linear.weight, 0, 0.01);
                        nn.init.zeros_(linear.bias);
                    }
                }
            }

            public override Tensor forward(Tensor x)
            {
                x = this.features.call(x);
                x = this.avgpool.call(x);
                x = torch.flatten(x, 1);
                x = this.classifier.call(x);
                return x;
            }
        }
    }

    public static partial class torchvision
    {
        public static partial class models
        {
            private static (MobileNetV3.InvertedResidualConfig[], long) _mobilenet_v3_conf(
                string arch,
                double width_mult = 1.0,
                bool reduced_tail = false,
                bool dilated = false)
            {
                long last_channel;
                MobileNetV3.InvertedResidualConfig[] inverted_residual_setting;
                var reduce_divider = reduced_tail ? 2 : 1;
                var dilation = dilated ? 2 : 1;
                MobileNetV3.InvertedResidualConfig bneck_conf(
                    long input_channels,
                    long kernel,
                    long expanded_channels,
                    long out_channels,
                    bool use_se,
                    string activation,
                    long stride,
                    long dilation) => new MobileNetV3.InvertedResidualConfig(
                        input_channels,
                        kernel,
                        expanded_channels,
                        out_channels,
                        use_se,
                        activation,
                        stride,
                        dilation,
                        width_mult: width_mult);
                long adjust_channels(long channels) =>
                    MobileNetV3.InvertedResidualConfig.adjust_channels(
                        channels,
                        width_mult: width_mult);
                if (arch == "mobilenet_v3_large") {
                    inverted_residual_setting = new MobileNetV3.InvertedResidualConfig[] {
                    bneck_conf(16, 3, 16, 16, false, "RE", 1, 1),
                    bneck_conf(16, 3, 64, 24, false, "RE", 2, 1),
                    bneck_conf(24, 3, 72, 24, false, "RE", 1, 1),
                    bneck_conf(24, 5, 72, 40, true, "RE", 2, 1),
                    bneck_conf(40, 5, 120, 40, true, "RE", 1, 1),
                    bneck_conf(40, 5, 120, 40, true, "RE", 1, 1),
                    bneck_conf(40, 3, 240, 80, false, "HS", 2, 1),
                    bneck_conf(80, 3, 200, 80, false, "HS", 1, 1),
                    bneck_conf(80, 3, 184, 80, false, "HS", 1, 1),
                    bneck_conf(80, 3, 184, 80, false, "HS", 1, 1),
                    bneck_conf(80, 3, 480, 112, true, "HS", 1, 1),
                    bneck_conf(112, 3, 672, 112, true, "HS", 1, 1),
                    bneck_conf(112, 5, 672, 160 / reduce_divider, true, "HS", 2, dilation),
                    bneck_conf(160 / reduce_divider, 5, 960 / reduce_divider, 160 / reduce_divider, true, "HS", 1, dilation),
                    bneck_conf(160 / reduce_divider, 5, 960 / reduce_divider, 160 / reduce_divider, true, "HS", 1, dilation)
                };
                    last_channel = adjust_channels(1280 / reduce_divider);
                } else if (arch == "mobilenet_v3_small") {
                    inverted_residual_setting = new MobileNetV3.InvertedResidualConfig[] {
                    bneck_conf(16, 3, 16, 16, true, "RE", 2, 1),
                    bneck_conf(16, 3, 72, 24, false, "RE", 2, 1),
                    bneck_conf(24, 3, 88, 24, false, "RE", 1, 1),
                    bneck_conf(24, 5, 96, 40, true, "HS", 2, 1),
                    bneck_conf(40, 5, 240, 40, true, "HS", 1, 1),
                    bneck_conf(40, 5, 240, 40, true, "HS", 1, 1),
                    bneck_conf(40, 5, 120, 48, true, "HS", 1, 1),
                    bneck_conf(48, 5, 144, 48, true, "HS", 1, 1),
                    bneck_conf(48, 5, 288, 96 / reduce_divider, true, "HS", 2, dilation),
                    bneck_conf(96 / reduce_divider, 5, 576 / reduce_divider, 96 / reduce_divider, true, "HS", 1, dilation),
                    bneck_conf(96 / reduce_divider, 5, 576 / reduce_divider, 96 / reduce_divider, true, "HS", 1, dilation)
                };
                    last_channel = adjust_channels(1024 / reduce_divider);
                } else {
                    throw new ArgumentException($"Unsupported model type {arch}");
                }
                return (inverted_residual_setting, last_channel);
            }

            private static Modules.MobileNetV3 _mobilenet_v3(
                MobileNetV3.InvertedResidualConfig[] inverted_residual_setting,
                long last_channel)
            {
                var model = new MobileNetV3("MobileNetV3", inverted_residual_setting, last_channel);
                return model;
            }

            /// <summary>
            /// Constructs a large MobileNetV3 architecture from
            /// `Searching for MobileNetV3 https://arxiv.org/abs/1905.02244`.
            /// </summary>
            /// <returns></returns>
            public static MobileNetV3 mobilenet_v3_large()
            {
                var (inverted_residual_setting, last_channel) = _mobilenet_v3_conf("mobilenet_v3_large");
                return _mobilenet_v3(inverted_residual_setting, last_channel);
            }

            /// <summary>
            /// Constructs a small MobileNetV3 architecture from
            /// `Searching for MobileNetV3 https://arxiv.org/abs/1905.02244`.
            /// </summary>
            /// <returns></returns>
            public static MobileNetV3 mobilenet_v3_small()
            {
                var (inverted_residual_setting, last_channel) = _mobilenet_v3_conf("mobilenet_v3_small");
                return _mobilenet_v3(inverted_residual_setting, last_channel);
            }
        }
    }
}