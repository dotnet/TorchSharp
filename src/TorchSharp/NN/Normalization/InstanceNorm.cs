// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.PInvoke.NativeMethods;
#nullable enable
namespace TorchSharp
{
    using System.Globalization;
    using System.Transactions;
    using Modules;
    using TorchSharp.Utils;
    using F = TorchSharp.torch.nn.functional;

    namespace Modules
    {
        public abstract class InstanceNorm : NormBase
        {
            public InstanceNorm(long num_features, 
                                double eps, 
                                double? momentum, 
                                bool affine, 
                                bool track_running_stats,
                                Device? device, 
                                ScalarType? dtype, 
                                string name) : base(num_features, eps, momentum.HasValue ? momentum : 0.1, affine, track_running_stats, device, dtype, name) 
            {                
            }

            protected abstract long GetNumberOfBatchDimensions();

            public override Tensor forward(Tensor input)
            {
                ValidateInputDimensions(input);

                var feature_dim = (int)(input.ndim - GetNumberOfBatchDimensions());

                if (input.size((int)feature_dim) != num_features) {
                    throw new ArgumentException($"expected input's size at dim={feature_dim} to match num_features ({this.num_features}), but got: {input.size(feature_dim)}.");
                }

                if (feature_dim == 0) {
                    using var t0 = input.unsqueeze(0);
                    return ApplyInstanceNorm(t0).squeeze_(0);
                }
                else {
                    return ApplyInstanceNorm(input);
                }
            }

            private Tensor ApplyInstanceNorm(Tensor input)
            {
                return F.instance_norm(input, running_mean, running_var, weight, bias, training || !track_running_stats, momentum.HasValue ? momentum.Value : 0.1, eps);
            }
        }
    }
}