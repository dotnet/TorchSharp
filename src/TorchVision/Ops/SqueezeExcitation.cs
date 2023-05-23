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
using System.Linq;
using static TorchSharp.torch;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class ops
        {
            /// <summary>
            /// This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507
            /// Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.
            /// </summary>
            /// <param name="input_channels">Number of channels in the input image</param>
            /// <param name="squeeze_channels">Number of squeeze channels</param>
            /// <param name="activation">`delta` activation.</param>
            /// <param name="scale_activation">`sigma` activation</param>
            public static SqueezeExcitation SqueezeExcitation(
                long input_channels,
                long squeeze_channels,
                Func<nn.Module<Tensor, Tensor>>? activation = null,
                Func<nn.Module<Tensor, Tensor>>? scale_activation = null) => new SqueezeExcitation(input_channels, squeeze_channels, activation, scale_activation);
        }

        /// <summary>
        /// This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507
        /// Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.
        /// </summary>
        public class SqueezeExcitation : torch.nn.Module<Tensor, Tensor>
        {
            private readonly nn.Module<Tensor, Tensor> avgpool;
            private readonly nn.Module<Tensor, Tensor> fc1;
            private readonly nn.Module<Tensor, Tensor> fc2;
            private readonly nn.Module<Tensor, Tensor> activation;
            private readonly nn.Module<Tensor, Tensor> scale_activation;

            private long input_channels;

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="input_channels">Number of channels in the input image</param>
            /// <param name="squeeze_channels">Number of squeeze channels</param>
            /// <param name="activation">``delta`` activation</param>
            /// <param name="scale_activation">``sigma`` activation.</param>
            public SqueezeExcitation(
                long input_channels,
                long squeeze_channels,
                Func<nn.Module<Tensor, Tensor>>? activation = null,
                Func<nn.Module<Tensor, Tensor>>? scale_activation = null) : base(nameof(SqueezeExcitation))
            {
                this.input_channels = input_channels;

                this.avgpool = torch.nn.AdaptiveAvgPool2d(1);
                this.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1);
                this.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1);
                this.activation = activation is null ? nn.ReLU() : activation();
                this.scale_activation = scale_activation is null ? nn.Sigmoid() : scale_activation();

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
                if ((input.ndim == 4 && input.shape[1] == input_channels) ||
                    (input.ndim == 3 && input.shape[0] == input_channels)) {
                    using var _ = NewDisposeScope();
                    var scale = this._scale(input);
                    return (scale * input).MoveToOuterDisposeScope();
                }
                throw new ArgumentException("Expected 3D (unbatched) or 4D (batched) input to SqueezeExcitation");
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    avgpool.Dispose();
                    fc1.Dispose();
                    fc2.Dispose();
                    activation.Dispose();
                    scale_activation.Dispose();
                }
                base.Dispose(disposing);
            }
        }
    }
}
