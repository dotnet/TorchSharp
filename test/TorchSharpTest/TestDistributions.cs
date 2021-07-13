// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

using static TorchSharp.torch;
using static TorchSharp.torch.distributions;

using Xunit;

#nullable enable

namespace TorchSharp
{
    public class TestDistributions
    {
        [Fact]
        public void TestUniform1()
        {
            var dist = Uniform(torch.tensor(0.0), torch.tensor(3.5));
            {
                var sample = dist.sample();

                Assert.Empty(sample.shape);
                Assert.True(sample.ToDouble() >= 0.0 && sample.ToDouble() < 3.5);
            }
            {
                var sample = dist.sample(2,3);

                Assert.Equal(new long[] { 2, 3 }, sample.shape);
                Assert.All<double>(sample.Data<double>().ToArray(), d => Assert.True(d >= 0 && d < 3.5));
            }
            {
                var sample = dist.expand(new long[] { 3, 4 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 4 }, sample.shape);
                Assert.All<double>(sample.Data<double>().ToArray(), d => Assert.True(d >= 0 && d < 3.5));
            }
        }

        [Fact]
        public void TestBernoulli()
        {
            var dist = Bernoulli(torch.rand(3, dtype: ScalarType.Float64));
            {
                var sample = dist.sample();

                Assert.Single(sample.shape);
                Assert.All<double>(sample.Data<double>().ToArray(), d => Assert.True(d == 0 || d == 1));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
                Assert.All<double>(sample.Data<double>().ToArray(), d => Assert.True(d == 0 || d == 1));
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
                Assert.All<double>(sample.Data<double>().ToArray(), d => Assert.True(d == 0 || d == 1));
            }
        }

        [Fact]
        public void TestBinomial()
        {
            var dist = Binomial(torch.tensor(100), torch.rand(3, dtype: ScalarType.Float64));
            {
                var sample = dist.sample();

                Assert.Single(sample.shape);
                Assert.All<double>(sample.Data<double>().ToArray(), d => Assert.True(d >= 0 && d < 100));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
                Assert.All<double>(sample.Data<double>().ToArray(), d => Assert.True(d >= 0 && d < 100));
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
                Assert.All<double>(sample.Data<double>().ToArray(), d => Assert.True(d >= 0 && d < 100));
            }
        }
    }
}
