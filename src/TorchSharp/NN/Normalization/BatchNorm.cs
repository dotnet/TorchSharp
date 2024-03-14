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
        public abstract class BatchNorm : NormBase
        {
            public BatchNorm(long num_features, 
                            double eps, 
                            double momentum, 
                            bool affine, 
                            bool track_running_stats, 
                            Device? device, 
                            ScalarType? dtype, 
                            string name) : base(num_features, eps, momentum, affine, track_running_stats, device, dtype, name) 
            {                
            }

            public override Tensor forward(Tensor input)
            {
                ValidateInputDimensions(input);

                var exponential_average_factor = this.momentum;

                if (training && track_running_stats)
                {
                    if (num_batches_tracked is not null)
                    {
                        num_batches_tracked.add_(1);
                        exponential_average_factor = momentum == 0 ? (1.0 / (double)num_batches_tracked) : momentum;
                    }
                }

                var bn_training = training ? true : running_mean is not null && running_var is not null;
                var pr = !training || track_running_stats;

                return F.batch_norm(input, pr ? running_mean : null, pr ? running_var : null, weight, bias, bn_training, exponential_average_factor, eps);
            }

        }
    }
}