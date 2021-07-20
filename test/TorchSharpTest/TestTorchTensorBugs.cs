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
                var scalar = Float32Tensor.from(3.14f, torch.CUDA);
                Assert.Throws<InvalidOperationException>(() => scalar.DataItem<float>());
                var tensor = Float32Tensor.zeros(new long[] { 10, 10 }, torch.CUDA);
                Assert.Throws<InvalidOperationException>(() => tensor.Data<float>());
                Assert.Throws<InvalidOperationException>(() => tensor.Bytes());
            }
        }

        class DoubleIt : nn.CustomModule
        {
            public DoubleIt() : base("double") { }

            public override Tensor forward(Tensor t) => t * 2;
        }

        [Fact]
        public void ValidateIssue315()
        {
            // https://github.com/xamarin/TorchSharp/issues/315
            // custom module crash in GC thread

            // make Torch call our custom module by adding a ReLU in front of it
            using var net = nn.Sequential(
                ("relu", nn.ReLU()),
                ("double", new DoubleIt())
            );

            using var @in = Float32Tensor.from(3);
            using var @out = net.forward(@in);
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
    }
}