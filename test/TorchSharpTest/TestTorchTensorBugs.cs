// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

#nullable enable

namespace TorchSharp
{
    // The tests in this file are all derived from reported GitHub Issues, serving
    // as regression tests.

    public class TestTorchTensorBugs
    {
        [Fact]
        public void ValidateIssue145()
        {
            // Tensor.DataItem gives a hard crash on GPU tensor

            if (torch.cuda.is_available()) {
                var scalar = Float32Tensor.from(3.14f, torch.device.CUDA);
                Assert.Throws<InvalidOperationException>(() => scalar.DataItem<float>());
                var tensor = Float32Tensor.zeros(new long[] { 10, 10 }, torch.device.CUDA);
                Assert.Throws<InvalidOperationException>(() => tensor.Data<float>());
                Assert.Throws<InvalidOperationException>(() => tensor.Bytes());
            }
        }
    }
}