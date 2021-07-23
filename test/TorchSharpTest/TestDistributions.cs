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
        public void TestNormal()
        {
            var dist = Normal(torch.tensor(0.0), torch.tensor(3.5));
            {
                var sample = dist.sample();

                Assert.Empty(sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 4 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 4 }, sample.shape);
            }
        }

        [Fact]
        public void TestPoisson()
        {
            var dist = Poisson(torch.tensor(0.5));
            {
                var sample = dist.sample();

                Assert.Empty(sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 4 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 4 }, sample.shape);
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
        public void TestBeta()
        {
            var dist = Beta(torch.rand(3, 3) * 0.5f, torch.tensor(0.5f));
            {
                var sample = dist.sample();
                var entropy = dist.entropy();

                Assert.Equal(new long[] { 3, 3 }, sample.shape);
                Assert.Equal(new long[] { 3, 3 }, entropy.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestBinomial()
        {
            var dist = Binomial(torch.tensor(100), torch.rand(3, dtype: ScalarType.Float64));
            {
                var sample = dist.sample();

                Assert.Single(sample.shape);
                Assert.All<double>(sample.Data<double>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
                Assert.All<double>(sample.Data<double>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
                Assert.All<double>(sample.Data<double>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
        }


        [Fact]
        public void TestCategorical()
        {
            var categories = 7;
            var dist = Categorical(torch.rand(3, categories, dtype: ScalarType.Float64));
            {
                var sample = dist.sample();

                Assert.Single(sample.shape);
                Assert.Equal(3, sample.shape[0]);
                Assert.All<long>(sample.Data<long>().ToArray(), l => Assert.True(l >= 0 && l < categories));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
                Assert.All<long>(sample.Data<long>().ToArray(), l => Assert.True(l >= 0 && l < categories));
            }
        }


        [Fact]
        public void TestCauchy()
        {
            var dist = Cauchy(torch.rand(3, 3), torch.tensor(1.0f));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3, 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestChi2()
        {
            var dist = Chi2(torch.rand(3, 3));
            {
                var sample = dist.sample();
                var entropy = dist.entropy();

                Assert.Equal(new long[] { 3, 3 }, sample.shape);
                Assert.Equal(new long[] { 3, 3 }, entropy.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestDirichlet()
        {
            var dist = Dirichlet(torch.rand(3, 3) * 0.5f);
            {
                var sample = dist.sample();
                var entropy = dist.entropy();

                Assert.Equal(new long[] { 3, 3 }, sample.shape);
                Assert.Equal(new long[] { 3 }, entropy.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestExponential()
        {
            var dist = Exponential(torch.rand(3, 3) * 0.5f);
            {
                var sample = dist.sample();
                var entropy = dist.entropy();

                Assert.Equal(new long[] { 3, 3 }, sample.shape);
                Assert.Equal(new long[] { 3, 3 }, entropy.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestFisherSnedecor()
        {
            var dist = FisherSnedecor(torch.rand(3, 3) * 1.5f, torch.tensor(2.0f));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3, 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestGamma()
        {
            var dist = Gamma(torch.rand(3, 3), torch.tensor(1.0f));
            {
                var sample = dist.sample();
                var entropy = dist.entropy();

                Assert.Equal(new long[] { 3, 3 }, sample.shape);
                Assert.Equal(new long[] { 3, 3 }, entropy.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestGeometric()
        {
            var dist = Geometric(torch.rand(3, 3));
            {
                var sample = dist.sample();
                var entropy = dist.entropy();

                Assert.Equal(new long[] { 3, 3 }, sample.shape);
                Assert.Equal(new long[] { 3, 3 }, entropy.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestMultinomial()
        {
            var categories = 17;
            var dist = Multinomial(100, torch.rand(3, categories));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3, categories }, sample.shape);
                Assert.All<float>(sample.Data<float>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, categories }, sample.shape);
                Assert.All<float>(sample.Data<float>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, categories }, sample.shape);
                Assert.All<float>(sample.Data<float>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
        }
    }
}
