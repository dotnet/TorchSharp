// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/a4f53308b2d0f1aa9191686e326f45c26053f686/torchvision/ops/misc.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using static TorchSharp.torch;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class ops
        {
            private static nn.Module<Tensor, Tensor> ConvNormActivation(
                long in_channels,
                long out_channels,
                long kernel_size = 3,
                long stride = 1,
                long? padding = null,
                long groups = 1,
                Func<long, nn.Module<Tensor, Tensor>>? norm_layer = null,
                Func<bool, nn.Module<Tensor, Tensor>>? activation_layer = null,
                long dilation = 1,
                bool inplace = true,
                bool? bias = null,
                int rank = 2)
            {
                if (padding == null) {
                    padding = (kernel_size - 1) / 2 * dilation;
                }

                if (bias == null) {
                    bias = norm_layer == null;
                }

                var layers = new List<nn.Module<Tensor, Tensor>>();
                if (rank == 2) {
                    layers.Add(
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernelSize: kernel_size,
                            stride: stride,
                            padding: padding.Value,
                            dilation: dilation,
                            groups: groups,
                            bias: bias.Value));
                } else if (rank == 3) {
                    layers.Add(
                        nn.Conv3d(
                            in_channels,
                            out_channels,
                            kernelSize: kernel_size,
                            stride: stride,
                            padding: padding.Value,
                            dilation: dilation,
                            groups: groups,
                            bias: bias.Value));
                } else {
                    throw new ArgumentOutOfRangeException("rank must be 2 or 3.");
                }

                if (norm_layer != null) {
                    layers.Add(norm_layer(out_channels));
                }

                if (activation_layer != null) {
                    layers.Add(activation_layer(inplace));
                }
                return nn.Sequential(layers);
            }

            /// <summary>
            /// Configurable block used for Convolution2d-Normalization-Activation blocks.
            /// </summary>
            /// <param name="in_channels">Number of channels in the input image</param>
            /// <param name="out_channels">Number of channels produced by the Convolution-Normalization-Activation block</param>
            /// <param name="kernel_size">Size of the convolving kernel.</param>
            /// <param name="stride">Stride of the convolution.</param>
            /// <param name="padding">Padding added to all four sides of the input. Default: null, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels.</param>
            /// <param name="norm_layer">Norm layer that will be stacked on top of the convolution layer. If ``null`` this layer wont be used.</param>
            /// <param name="activation_layer">Activation function which will be stacked on top of the normalization layer (if not null), otherwise on top of the conv layer. If ``null`` this layer wont be used.</param>
            /// <param name="dilation">Spacing between kernel elements.</param>
            /// <param name="inplace">Parameter for the activation layer, which can optionally do the operation in-place.</param>
            /// <param name="bias">Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is null``.</param>
            public static nn.Module<Tensor, Tensor> Conv2dNormActivation(
                long in_channels,
                long out_channels,
                long kernel_size = 3,
                long stride = 1,
                long? padding = null,
                long groups = 1,
                Func<long, nn.Module<Tensor, Tensor>>? norm_layer = null,
                Func<bool, nn.Module<Tensor, Tensor>>? activation_layer = null,
                long dilation = 1,
                bool inplace = true,
                bool? bias = null)
            {
                return ConvNormActivation(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups,
                    norm_layer,
                    activation_layer,
                    dilation,
                    inplace,
                    bias,
                    rank: 2);
            }

            /// <summary>
            /// Configurable block used for Convolution3d-Normalization-Activation blocks.
            /// </summary>
            /// <param name="in_channels">Number of channels in the input image</param>
            /// <param name="out_channels">Number of channels produced by the Convolution-Normalization-Activation block</param>
            /// <param name="kernel_size">Size of the convolving kernel.</param>
            /// <param name="stride">Stride of the convolution.</param>
            /// <param name="padding">Padding added to all four sides of the input. Default: null, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels.</param>
            /// <param name="norm_layer">Norm layer that will be stacked on top of the convolution layer. If ``null`` this layer wont be used.</param>
            /// <param name="activation_layer">Activation function which will be stacked on top of the normalization layer (if not null), otherwise on top of the conv layer. If ``null`` this layer wont be used.</param>
            /// <param name="dilation">Spacing between kernel elements.</param>
            /// <param name="inplace">Parameter for the activation layer, which can optionally do the operation in-place.</param>
            /// <param name="bias">Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is null``.</param>
            public static nn.Module<Tensor, Tensor> Conv3dNormActivation(
                long in_channels,
                long out_channels,
                long kernel_size = 3,
                long stride = 1,
                long? padding = null,
                long groups = 1,
                Func<long, nn.Module<Tensor, Tensor>>? norm_layer = null,
                Func<bool, nn.Module<Tensor, Tensor>>? activation_layer = null,
                long dilation = 1,
                bool inplace = true,
                bool? bias = null)
            {
                return ConvNormActivation(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups,
                    norm_layer,
                    activation_layer,
                    dilation,
                    inplace,
                    bias,
                    rank: 3);
            }

            internal class SqueezeExcitation : torch.nn.Module<Tensor, Tensor>
            {
                private readonly nn.Module<Tensor, Tensor> avgpool;
                private readonly nn.Module<Tensor, Tensor> fc1;
                private readonly nn.Module<Tensor, Tensor> fc2;
                private readonly nn.Module<Tensor, Tensor> activation;
                private readonly nn.Module<Tensor, Tensor> scale_activation;

                /// <summary>
                /// This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
                /// Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.
                /// </summary>
                /// <param name="name"></param>
                /// <param name="input_channels">Number of channels in the input image</param>
                /// <param name="squeeze_channels">Number of squeeze channels</param>
                /// <param name="activation">``delta`` activation</param>
                /// <param name="scale_activation">``sigma`` activation.</param>
                public SqueezeExcitation(
                    string name,
                    long input_channels,
                    long squeeze_channels,
                    Func<nn.Module<Tensor, Tensor>> activation,
                    Func<nn.Module<Tensor, Tensor>> scale_activation) : base(name)
                {
                    this.avgpool = torch.nn.AdaptiveAvgPool2d(1);
                    this.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1);
                    this.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1);
                    this.activation = activation();
                    this.scale_activation = scale_activation();

                    RegisterComponents();
                }

                private Tensor _scale(Tensor input)
                {
                    var scale = this.avgpool.call(input);
                    scale = this.fc1.call(scale);
                    scale = this.activation.call(scale);
                    scale = this.fc2.call(scale);
                    return this.scale_activation.call(scale);
                }

                public override Tensor forward(Tensor input)
                {
                    var scale = this._scale(input);
                    return scale * input;
                }
            }
        }
    }
}