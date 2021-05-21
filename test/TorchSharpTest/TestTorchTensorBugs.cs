// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp;
using TorchSharp.Tensor;
using Xunit;

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
            // TorchTensor.DataItem gives a hard crash on GPU tensor

            if (Torch.IsCudaAvailable()) {
                var scalar = Float32Tensor.from(3.14f, Device.CUDA);
                Assert.Throws<InvalidOperationException>(() => scalar.DataItem<float>());
                var tensor = Float32Tensor.zeros(new long[] { 10, 10 }, Device.CUDA);
                Assert.Throws<InvalidOperationException>(() => tensor.Data<float>());
                Assert.Throws<InvalidOperationException>(() => tensor.Bytes());
            }
        }
    }
}