// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py
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
            /// MNASNet with depth multiplier of 0.5 from
            /// "MnasNet: Platform-Aware Neural Architecture Search for Mobile".
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
            /// model = models.mnasnet0_5(pretrained=True)
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
            public static Modules.MNASNet mnasnet0_5(int num_classes = 1000, float dropout = 0.2f, string? weights_file = null, bool skipfc = true, Device? device = null)
            {
                return new Modules.MNASNet(0.5, num_classes, dropout, weights_file, skipfc, device);
            }

            /// <summary>
            /// MNASNet with depth multiplier of 0.75 from
            /// "MnasNet: Platform-Aware Neural Architecture Search for Mobile".
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last linear layer of the classifier will not be loaded from the weights file.</param>
            /// <param name="device">The device to locate the model on.</param>
            public static Modules.MNASNet mnasnet0_75(int num_classes = 1000, float dropout = 0.2f, string? weights_file = null, bool skipfc = true, Device? device = null)
            {
                return new Modules.MNASNet(0.75, num_classes, dropout, weights_file, skipfc, device);
            }

            /// <summary>
            /// MNASNet with depth multiplier of 1.0 from
            /// "MnasNet: Platform-Aware Neural Architecture Search for Mobile".
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last linear layer of the classifier will not be loaded from the weights file.</param>
            /// <param name="device">The device to locate the model on.</param>
            public static Modules.MNASNet mnasnet1_0(int num_classes = 1000, float dropout = 0.2f, string? weights_file = null, bool skipfc = true, Device? device = null)
            {
                return new Modules.MNASNet(1.0, num_classes, dropout, weights_file, skipfc, device);
            }

            /// <summary>
            /// MNASNet with depth multiplier of 1.3 from
            /// "MnasNet: Platform-Aware Neural Architecture Search for Mobile".
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last linear layer of the classifier will not be loaded from the weights file.</param>
            /// <param name="device">The device to locate the model on.</param>
            public static Modules.MNASNet mnasnet1_3(int num_classes = 1000, float dropout = 0.2f, string? weights_file = null, bool skipfc = true, Device? device = null)
            {
                return new Modules.MNASNet(1.3, num_classes, dropout, weights_file, skipfc, device);
            }
        }
    }

    namespace Modules
    {
        // Based on https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py
        // License: https://github.com/pytorch/vision/blob/main/LICENSE

        /// <summary>
        /// MNASNet, as described in https://arxiv.org/abs/1807.11626.
        /// This implements the B1 variant of the model.
        /// </summary>
        public class MNASNet : Module<Tensor, Tensor>
        {
            // Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is 1.0 - tensorflow.
            private const double _BN_MOMENTUM = 1.0 - 0.9997;

            private class _InvertedResidual : Module<Tensor, Tensor>
            {
                private readonly bool apply_residual;
                private readonly Module<Tensor, Tensor> layers;

                public _InvertedResidual(string name, long in_ch, long out_ch, long kernel_size, long stride, long expansion_factor, double bn_momentum)
                    : base(name)
                {
                    if (stride != 1 && stride != 2)
                        throw new ArgumentOutOfRangeException($"stride should be 1 or 2 instead of {stride}");
                    if (kernel_size != 3 && kernel_size != 5)
                        throw new ArgumentOutOfRangeException($"kernel_size should be 3 or 5 instead of {kernel_size}");

                    var mid_ch = in_ch * expansion_factor;
                    apply_residual = in_ch == out_ch && stride == 1;
                    layers = Sequential(
                        // Pointwise
                        Conv2d(in_ch, mid_ch, 1, bias: false),
                        BatchNorm2d(mid_ch, momentum: bn_momentum),
                        ReLU(inplace: true),
                        // Depthwise
                        Conv2d(mid_ch, mid_ch, kernel_size, padding: kernel_size / 2, stride: stride, groups: mid_ch, bias: false),
                        BatchNorm2d(mid_ch, momentum: bn_momentum),
                        ReLU(inplace: true),
                        // Linear pointwise. Note that there's no activation.
                        Conv2d(mid_ch, out_ch, 1, bias: false),
                        BatchNorm2d(out_ch, momentum: bn_momentum)
                    );
                    RegisterComponents();
                }

                protected override void Dispose(bool disposing)
                {
                    if (disposing) {
                        layers.Dispose();
                    }
                    base.Dispose(disposing);
                }

                public override Tensor forward(Tensor input)
                {
                    if (apply_residual) {
                        return layers.call(input) + input;
                    } else {
                        return layers.call(input);
                    }
                }
            }

            /// <summary>
            /// Creates a stack of inverted residuals.
            /// </summary>
            private static Module<Tensor, Tensor> _stack(long in_ch, long out_ch, long kernel_size, long stride, long exp_factor, int repeats, double bn_momentum)
            {
                if (repeats < 1)
                    throw new ArgumentOutOfRangeException($"repeats should be >= 1, instead got {repeats}");

                var modules = new List<Module<Tensor, Tensor>>();
                // First one has no skip, because feature map size changes.
                modules.Add(new _InvertedResidual("_InvertedResidual", in_ch, out_ch, kernel_size, stride, exp_factor, bn_momentum));
                for (int i = 1; i < repeats; i++) {
                    modules.Add(new _InvertedResidual("_InvertedResidual", out_ch, out_ch, kernel_size, 1, exp_factor, bn_momentum));
                }
                return Sequential(modules);
            }

            /// <summary>
            /// Asymmetric rounding to make val divisible by divisor.
            /// With default bias, will round up, unless the number is no more than 10% greater
            /// than the smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88.
            /// </summary>
            private static int _round_to_multiple_of(double val, int divisor, double round_up_bias = 0.9)
            {
                if (round_up_bias <= 0.0 || round_up_bias >= 1.0)
                    throw new ArgumentOutOfRangeException($"round_up_bias should be greater than 0.0 and smaller than 1.0 instead of {round_up_bias}");
                var new_val = Math.Max(divisor, (int)(val + divisor / 2) / divisor * divisor);
                return new_val >= round_up_bias * val ? new_val : new_val + divisor;
            }

            /// <summary>
            /// Scales tensor depths as in reference MobileNet code, prefers rounding up rather than down.
            /// </summary>
            private static int[] _get_depths(double alpha)
            {
                var depths = new int[] { 32, 16, 24, 40, 80, 96, 192, 320 };
                var result = new int[depths.Length];
                for (int i = 0; i < depths.Length; i++) {
                    result[i] = _round_to_multiple_of(depths[i] * alpha, 8);
                }
                return result;
            }

            private readonly Module<Tensor, Tensor> layers;
            private readonly Module<Tensor, Tensor> classifier;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    layers.Dispose();
                    classifier.Dispose();
                }
                base.Dispose(disposing);
            }

            public MNASNet(double alpha, int num_classes = 1000, float dropout = 0.2f,
                string? weights_file = null, bool skipfc = true, Device? device = null)
                : base(nameof(MNASNet))
            {
                if (alpha <= 0.0)
                    throw new ArgumentOutOfRangeException($"alpha should be greater than 0.0 instead of {alpha}");

                var depths = _get_depths(alpha);
                var layerList = new List<Module<Tensor, Tensor>> {
                    // First layer: regular conv.
                    Conv2d(3, depths[0], 3, padding: 1, stride: 2, bias: false),
                    BatchNorm2d(depths[0], momentum: _BN_MOMENTUM),
                    ReLU(inplace: true),
                    // Depthwise separable, no skip.
                    Conv2d(depths[0], depths[0], 3, padding: 1, stride: 1, groups: depths[0], bias: false),
                    BatchNorm2d(depths[0], momentum: _BN_MOMENTUM),
                    ReLU(inplace: true),
                    Conv2d(depths[0], depths[1], 1, padding: 0L, stride: 1, bias: false),
                    BatchNorm2d(depths[1], momentum: _BN_MOMENTUM),
                    // MNASNet blocks: stacks of inverted residuals.
                    _stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM),
                    _stack(depths[2], depths[3], 5, 2, 3, 3, _BN_MOMENTUM),
                    _stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM),
                    _stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM),
                    _stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM),
                    _stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM),
                    // Final mapping to classifier input.
                    Conv2d(depths[7], 1280, 1, padding: 0L, stride: 1, bias: false),
                    BatchNorm2d(1280, momentum: _BN_MOMENTUM),
                    ReLU(inplace: true),
                };
                layers = Sequential(layerList);
                classifier = Sequential(
                    Dropout(p: dropout, inplace: true),
                    Linear(1280, num_classes)
                );

                RegisterComponents();

                // Weight initialization
                foreach (var (_, m) in named_modules()) {
                    if (m is Modules.Conv2d conv) {
                        init.kaiming_normal_(conv.weight, mode: init.FanInOut.FanOut);
                        if (conv.bias is not null)
                            init.zeros_(conv.bias);
                    } else if (m is Modules.BatchNorm2d norm) {
                        init.ones_(norm.weight);
                        init.zeros_(norm.bias);
                    } else if (m is Modules.Linear linear) {
                        init.kaiming_uniform_(linear.weight, mode: init.FanInOut.FanOut, nonlinearity: init.NonlinearityType.Sigmoid);
                        init.zeros_(linear.bias);
                    }
                }

                if (!string.IsNullOrEmpty(weights_file)) {
                    this.load(weights_file!, skip: skipfc ? new[] { "classifier.1.weight", "classifier.1.bias" } : null);
                }

                if (device != null && device.type != DeviceType.CPU)
                    this.to(device);
            }

            public override Tensor forward(Tensor x)
            {
                using (var _ = NewDisposeScope()) {
                    x = layers.call(x);
                    // Equivalent to global avgpool and removing H and W dimensions.
                    x = x.mean(new long[] { 2, 3 });
                    return classifier.call(x).MoveToOuterDisposeScope();
                }
            }
        }
    }
}
