// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/6ebbdfe8a6b47e1f6d6164b0c86ac48839281602/torchvision/models/_utils.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

using System;

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