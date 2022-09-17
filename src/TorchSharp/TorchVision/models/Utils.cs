// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;

using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class models
        {
            internal static partial class _utils
            {
                /// <summary>
                /// This function is taken from the original tf repo.
                /// It ensures that all layers have a channel number that is divisible by 8
                /// It can be seen here:
                /// https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
                /// </summary>
                internal static long _make_divisible(double v, long divisor, long? min_value = null)
                {
                    if (!min_value.HasValue) {
                        min_value = divisor;
                    }
                    var new_v = Math.Max(min_value.Value, (long)(v + divisor / 2.0) / divisor * divisor);
                    // Make sure that round down does not go down by more than 10%.
                    if (new_v < 0.9 * v) {
                        new_v += divisor;
                    }
                    return new_v;
                }
            }
        }
    }
}