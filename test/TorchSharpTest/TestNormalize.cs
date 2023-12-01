// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Linq;
using TorchSharp.Modules;
using Xunit;

namespace TorchSharp
{
    using static torch;
    using static torch.nn;

    public class TestNormalize
    {
        [Fact]
        public void TestLayerNorm()
        {
            var precomputedOutput = torch.tensor(new[, ,]
            {{{-0.6475, -0.9482, -0.9020, -0.5090,  0.2309},
             { 1.3176,  2.7512,  4.5316,  6.6588,  9.1329},
             {11.9538, 15.1216, 18.6362, 22.4977, 26.7060},
             {31.2611, 36.1631, 41.4120, 47.0076, 52.9502}},

            {{-0.6475, -0.9482, -0.9020, -0.5090,  0.2309},
             { 1.3176,  2.7512,  4.5316,  6.6588,  9.1329},
             {11.9538, 15.1216, 18.6362, 22.4977, 26.7060},
             {31.2611, 36.1631, 41.4120, 47.0076, 52.9502}},

            {{-0.6475, -0.9482, -0.9020, -0.5090,  0.2309},
             { 1.3176,  2.7512,  4.5316,  6.6588,  9.1329},
             {11.9538, 15.1216, 18.6362, 22.4977, 26.7060},
             {31.2611, 36.1631, 41.4120, 47.0076, 52.9502}}
            }, dtype: torch.float32);

            var input_data = torch.arange(1, 61, dtype: torch.float32).reshape(3, 4, 5);

            var normalized_shape = new long[] { 4, 5 };
            var weight = torch.arange(1, 21, dtype: torch.float32).reshape(4, 5);
            var bias = torch.arange(1, 21, dtype: torch.float32).reshape(4, 5);
            var eps = 1e-5;

            var output = torch.nn.functional.layer_norm(input_data, normalized_shape, weight, bias, eps);

            Assert.True(torch.allclose(output, precomputedOutput, atol: 1e-4));

            var layerNorm = torch.nn.LayerNorm(normalized_shape, eps, true, true);
            layerNorm.weight = new Parameter(weight);
            layerNorm.bias = new Parameter(bias);
            output = layerNorm.forward(input_data);

            Assert.True(torch.allclose(output, precomputedOutput, atol: 1e-4));
        }
    }
}

