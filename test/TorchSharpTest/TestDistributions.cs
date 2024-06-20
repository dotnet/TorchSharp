// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;
using static TorchSharp.torch.distributions;

using Xunit;
using System;

#nullable enable

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestDistributions
    {
        [Fact]
        public void TestUniform()
        {
            var dist = Uniform(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(3.5));
            Assert.True(torch.tensor(new[] { 2, 1.875, 1.825, }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.75, 0.88020833, 0.93520833 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 1.0986123, 1.178655, 1.2089603 }).allclose(dist.entropy(), rtol: 1e-3, atol: 1e-4));
            Assert.True(torch.tensor(new[] { 0.16666667, 0.2307692308, 0.2537313433 }).allclose(dist.cdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { 3.5000, 3.5000, 3.5000 }).allclose(dist.icdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { -1.0986123, -1.178655, -1.2089603 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d < 3.5));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d < 3.5));
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d < 3.5));
            }
        }

        [Fact]
        public void TestUniformGen()
        {
            var gen = new Generator(4711);

            var dist = Uniform(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(3.5), gen);
            Assert.True(torch.tensor(new[] { 2, 1.875, 1.825, }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.75, 0.88020833, 0.93520833 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 1.0986123, 1.178655, 1.2089603 }).allclose(dist.entropy(), rtol: 1e-3, atol: 1e-4));
            Assert.True(torch.tensor(new[] { 0.16666667, 0.2307692308, 0.2537313433 }).allclose(dist.cdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { 3.5000, 3.5000, 3.5000 }).allclose(dist.icdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { -1.0986123, -1.178655, -1.2089603 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d < 3.5));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d < 3.5));
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d < 3.5));
            }
        }

        [Fact]
        public void TestNormal()
        {
            var dist = Normal(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }), torch.tensor(new[] { 0.15, 0.05, 0.25 }));
            Assert.True(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.0225, 0.0025, 0.0625 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }).allclose(dist.mode));

            Assert.True(torch.tensor(new[] { -4.5774, -110.4232, -26.9126 }).allclose(dist.log_prob(torch.arange(3))));
            Assert.True(torch.tensor(new[] { -0.478181, -1.576794, 0.032644 }).allclose(dist.entropy(), rtol: 1e-3, atol: 1e-4));

            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestNormalGen()
        {
            var gen = new Generator(4711);

            var dist = Normal(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }), torch.tensor(new[] { 0.15, 0.05, 0.25 }), gen);
            Assert.True(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.0225, 0.0025, 0.0625 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }).allclose(dist.mode));

            Assert.True(torch.tensor(new[] { -4.5774, -110.4232, -26.9126 }).allclose(dist.log_prob(torch.arange(3))));
            Assert.True(torch.tensor(new[] { -0.478181, -1.576794, 0.032644 }).allclose(dist.entropy()));

            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestHalfNormal()
        {
            var dist = HalfNormal(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }));
            Assert.True(torch.tensor(new[] { 0.39894228, 0.19947114, 0.11968268 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.0, 0.0, 0.0 }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { 0.090845057, 0.022711264, 0.0081760551 }).allclose(dist.variance));

            Assert.True(torch.tensor(new[] { 0.9544997, 0.9999366, 0.99999999 }).allclose(dist.cdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { 0.032644172, -0.66050301, -1.1713286 }).allclose(dist.entropy()));
            Assert.True(torch.tensor(new[] { -1.5326442, -6.839497, -20.550894 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestHalfNormalGen()
        {
            var dist = HalfNormal(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }), new Generator(4711));
            Assert.True(torch.tensor(new[] { 0.39894228, 0.19947114, 0.11968268 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.0, 0.0, 0.0 }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { 0.090845057, 0.022711264, 0.0081760551 }).allclose(dist.variance));

            Assert.True(torch.tensor(new[] { 0.9544997, 0.9999366, 0.99999999 }).allclose(dist.cdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { 0.032644172, -0.66050301, -1.1713286 }).allclose(dist.entropy()));
            Assert.True(torch.tensor(new[] { -1.5326442, -6.839497, -20.550894 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestHalfCauchy()
        {
            var dist = HalfCauchy(torch.tensor(new[] { 0.5, 0.25, 0.15 }));
            Assert.True(torch.tensor(new[] { 0.0, 0.0, 0.0 }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { 0.70483276, 0.84404174, 0.90521372 }).allclose(dist.cdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { -11438666, -5719333, -3431599.8 }).allclose(dist.icdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { 1.1447299, 0.45158271, -0.059242918 }).allclose(dist.entropy()));
            Assert.True(torch.tensor(new[] { -1.3678734, -1.8985017, -2.3709533 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestHalfCauchyGen()
        {
            var dist = HalfCauchy(torch.tensor(new[] { 0.5, 0.25, 0.15 }), new Generator(4711));
            Assert.True(torch.tensor(new[] { 0.0, 0.0, 0.0 }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { 0.70483276, 0.84404174, 0.90521372 }).allclose(dist.cdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { -11438666, -5719333, -3431599.8 }).allclose(dist.icdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { 1.1447299, 0.45158271, -0.059242918 }).allclose(dist.entropy()));
            Assert.True(torch.tensor(new[] { -1.3678734, -1.8985017, -2.3709533 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestPareto()
        {
            var dist = Pareto(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(1.0));
            Assert.True(torch.tensor(new[] { 0.5, 0.25, 0.15, }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { 0.5000, 0.7500, 0.8500 }).allclose(dist.cdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { 1.3068528, 0.61370564, 0.10288002 }).allclose(dist.entropy()));
            Assert.True(torch.tensor(new[] { -0.69314718, -1.3862944, -1.89712 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestParetoGen()
        {
            var dist = Pareto(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(1.0), new Generator(4711));
            Assert.True(torch.tensor(new[] { 0.5, 0.25, 0.15, }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { 0.5000, 0.7500, 0.8500 }).allclose(dist.cdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { 1.3068528, 0.61370564, 0.10288002 }).allclose(dist.entropy()));
            Assert.True(torch.tensor(new[] { -0.69314718, -1.3862944, -1.89712 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestLogNormal()
        {
            var dist = LogNormal(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(0.5));
            Assert.True(torch.tensor(new[] { 1.868246, 1.4549914, 1.3165307 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 1.2840254, 1, 0.90483742 }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { 0.99134611, 0.60128181, 0.49228791 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 1.2257914, 0.97579135, 0.87579135 }).allclose(dist.entropy()));
            Assert.True(torch.tensor(new[] { 0.15865525, 0.308537538, 0.38208857 }).allclose(dist.cdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity }).allclose(dist.icdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { -0.72579135, -0.35079135, -0.27079135 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestLogNormalGen()
        {
            var dist = LogNormal(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(0.5), new Generator(4711));
            Assert.True(torch.tensor(new[] { 1.868246, 1.4549914, 1.3165307 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 1.2840254, 1, 0.90483742 }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { 0.99134611, 0.60128181, 0.49228791 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 1.2257914, 0.97579135, 0.87579135 }).allclose(dist.entropy()));
            Assert.True(torch.tensor(new[] { 0.15865525, 0.308537538, 0.38208857 }).allclose(dist.cdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity }).allclose(dist.icdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { -0.72579135, -0.35079135, -0.27079135 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestGumbel()
        {
            var dist = Gumbel(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(3.5));
            Assert.True(torch.tensor(new[] { 2.5202548, 2.2702548, 2.1702548 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.5, 0.25, 0.15 }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { 20.150442, 20.150442, 20.150442 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 2.8299786, 2.8299786, 2.8299786 }).allclose(dist.entropy()));
            Assert.True(torch.tensor(new[] { 0.42026160, 0.44614211, 0.456400959 }).allclose(dist.cdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { -2.26249801, -2.274166429, -2.28000367 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestGumbelGen()
        {
            var dist = Gumbel(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(3.5), new Generator(4711));
            Assert.True(torch.tensor(new[] { 2.5202548, 2.2702548, 2.1702548 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.5, 0.25, 0.15 }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { 20.150442, 20.150442, 20.150442 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 2.8299786, 2.8299786, 2.8299786 }).allclose(dist.entropy()));
            Assert.True(torch.tensor(new[] { 0.42026160, 0.44614211, 0.456400959 }).allclose(dist.cdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { -2.26249801, -2.274166429, -2.28000367 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestLaplace()
        {
            var dist = Laplace(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(0.5));
            Assert.True(torch.tensor(new[] { 0.5, 0.25, 0.15 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.5, 0.25, 0.15 }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { 0.5, 0.5, 0.5 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 1.0, 1.0, 1.0 }).allclose(dist.entropy()));
            Assert.True(torch.tensor(new[] { 0.81606028, 0.88843492, 0.90865824 }).allclose(dist.cdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity }).allclose(dist.icdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { -1, -1.5, -1.7 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestLaplaceGen()
        {
            var gen = new Generator(4711);

            var dist = Laplace(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(0.5), gen);
            Assert.True(torch.tensor(new[] { 0.5, 0.25, 0.15 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.5, 0.25, 0.15 }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { 0.5, 0.5, 0.5 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 1.0, 1.0, 1.0 }).allclose(dist.entropy()));
            Assert.True(torch.tensor(new[] { 0.81606028, 0.88843492, 0.90865824 }).allclose(dist.cdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity }).allclose(dist.icdf(torch.ones(3))));
            Assert.True(torch.tensor(new[] { -1, -1.5, -1.7 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestPoisson()
        {
            var dist = Poisson(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }));
            Assert.True(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 0.0, 0.0, 0.0 }).allclose(dist.mode));

            var log_prob = dist.log_prob(torch.arange(3));

            Assert.True(torch.tensor(new[] { -0.5000, -1.6363, -4.6374 }).allclose(log_prob));

            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestPoissonGen()
        {
            var gen = new Generator(4711);
            var dist = Poisson(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }), gen);
            Assert.True(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 0.0, 0.0, 0.0 }).allclose(dist.mode));

            var log_prob = dist.log_prob(torch.arange(3));

            Assert.True(torch.tensor(new[] { -0.5000, -1.6363, -4.6374 }).allclose(log_prob));

            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestBernoulli()
        {
            var dist = Bernoulli(torch.tensor(new[] { 0.5, 0.25, 0.15 }));
            Assert.True(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.2500, 0.1875, 0.1275 }).allclose(dist.variance, rtol: 1e-3, atol: 1e-4));
            Assert.True(torch.tensor(new[] { -0.69314718, -1.38629436, -1.897119984 }).allclose(dist.log_prob(torch.ones(3).to(torch.float64))));
            Assert.True(torch.tensor(new[] { 0.693147181, 0.5623351446, 0.4227090878 }).allclose(dist.entropy()));

            {
                var sample = dist.sample();

                Assert.Single(sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d == 0 || d == 1));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d == 0 || d == 1));
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d == 0 || d == 1));
            }
        }

        [Fact]
        public void TestBernoulliGen()
        {
            var gen = new Generator(4711);
            var dist = Bernoulli(torch.tensor(new[] { 0.5, 0.25, 0.15 }), generator: gen);
            Assert.True(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.2500, 0.1875, 0.1275 }).allclose(dist.variance, rtol: 1e-3, atol: 1e-4));
            Assert.True(torch.tensor(new[] { -0.69314718, -1.38629436, -1.897119984 }).allclose(dist.log_prob(torch.ones(3).to(torch.float64))));
            Assert.True(torch.tensor(new[] { 0.693147181, 0.5623351446, 0.4227090878 }).allclose(dist.entropy()));

            {
                var sample = dist.sample();

                Assert.Single(sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d == 0 || d == 1));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d == 0 || d == 1));
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d == 0 || d == 1));
            }
        }

        [Fact]
        public void TestLogitRelaxedBernoulli()
        {
            var temp = torch.rand(3, dtype: ScalarType.Float64);

            var dist = LogitRelaxedBernoulli(temp, torch.rand(3, dtype: ScalarType.Float64));
            {
                var sample = dist.sample();

                Assert.Single(sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestLogitRelaxedBernoulliGen()
        {
            var gen = new Generator(4711);
            var temp = torch.rand(3, dtype: ScalarType.Float64);

            var dist = LogitRelaxedBernoulli(temp, torch.rand(3, dtype: ScalarType.Float64), generator: gen);
            {
                var sample = dist.sample();

                Assert.Single(sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestRelaxedBernoulli()
        {
            var temp = torch.rand(3, dtype: ScalarType.Float64);

            var dist = RelaxedBernoulli(temp, torch.rand(3, dtype: ScalarType.Float64));
            {
                var sample = dist.sample();

                Assert.Single(sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestRelaxedBernoulliGen()
        {
            var gen = new Generator(4711);
            var temp = torch.rand(3, dtype: ScalarType.Float64);

            var dist = RelaxedBernoulli(temp, torch.rand(3, dtype: ScalarType.Float64), generator: gen);
            {
                var sample = dist.sample();

                Assert.Single(sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestBeta()
        {
            var dist = Beta(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(0.15f));
            Assert.True(torch.tensor(new[] { 0.769230762, 0.62499999, 0.5000 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 1.0, 1.0, 1.0 }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { 0.10758472, 0.16741072, 0.192307691 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { -3.0250008, -2.72106057, -3.4215678 }).allclose(dist.entropy()));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestBetaGen()
        {
            var gen = new Generator(4711);
            var dist = Beta(torch.rand(3, 3) * 0.5f, torch.tensor(0.5f), gen);
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
            var dist = Binomial(torch.tensor(100), torch.tensor(new[] { 0.25, 0.25, 0.15 }));
            Assert.True(torch.tensor(new[] { 25.0, 25.0, 15.0 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 18.7500, 18.7500, 12.7500 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 25.0, 25.0, 15.0 }).allclose(dist.mode));

            var log_prob = dist.log_prob(torch.arange(3));

            Assert.True(torch.tensor(new[] { -28.7682, -25.2616, -11.2140 }).allclose(log_prob));

            {
                var sample = dist.sample();

                Assert.Single(sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
        }

        [Fact]
        public void TestBinomialGen()
        {
            var gen = new Generator(4711);
            var dist = Binomial(torch.tensor(100), torch.tensor(new[] { 0.25, 0.25, 0.15 }), generator: gen);
            Assert.True(torch.tensor(new[] { 25.0, 25.0, 15.0 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 18.7500, 18.7500, 12.7500 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 25.0, 25.0, 15.0 }).allclose(dist.mode));

            var log_prob = dist.log_prob(torch.arange(3));

            Assert.True(torch.tensor(new[] { -28.7682, -25.2616, -11.2140 }).allclose(log_prob));

            {
                var sample = dist.sample();

                Assert.Single(sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
        }

        [Fact]
        public void TestNegativeBinomial()
        {
            var dist = NegativeBinomial(torch.tensor(100), torch.tensor(new[] { 0.25, 0.25, 0.15 }));
            Assert.True(torch.tensor(new[] { 33.3333, 33.3333, 17.6471 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 44.4444, 44.4444, 20.7612 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 33.0, 33.0, 17.0 }).allclose(dist.mode));

            var log_prob = dist.log_prob(torch.arange(3));

            Assert.True(torch.tensor(new[] { -28.7682, -25.5493, -11.5190 }).allclose(log_prob));

            {
                var sample = dist.sample();

                Assert.Single(sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
        }

        [Fact]
        public void TestCategorical()
        {
            var gen = new Generator(4711);
            var categories = 7;
            var dist = Categorical(torch.rand(3, categories, dtype: ScalarType.Float64));
            {
                var sample = dist.sample();

                Assert.Single(sample.shape);
                Assert.Equal(3, sample.shape[0]);
                Assert.All<long>(sample.data<long>().ToArray(), l => Assert.True(l >= 0 && l < categories));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
                Assert.All<long>(sample.data<long>().ToArray(), l => Assert.True(l >= 0 && l < categories));
            }
        }


        [Fact]
        public void TestCategoricalGen()
        {
            var gen = new Generator(4711);
            var categories = 7;
            var dist = Categorical(torch.rand(3, categories, dtype: ScalarType.Float64), generator: gen);
            {
                var sample = dist.sample();

                Assert.Single(sample.shape);
                Assert.Equal(3, sample.shape[0]);
                Assert.All<long>(sample.data<long>().ToArray(), l => Assert.True(l >= 0 && l < categories));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
                Assert.All<long>(sample.data<long>().ToArray(), l => Assert.True(l >= 0 && l < categories));
            }
        }

        [Fact]
        public void TestOneHotCategorical()
        {
            var categories = 7;
            var dist = OneHotCategorical(torch.rand(3, categories, dtype: ScalarType.Float64));
            {
                var sample = dist.sample();

                Assert.Equal(2, sample.ndim);
                Assert.Equal(3, sample.shape[0]);
                Assert.Equal(categories, sample.shape[1]);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(4, sample.ndim);
                Assert.Equal(new long[] { 2, 3, 3, 7 }, sample.shape);
            }
        }


        [Fact]
        public void TestOneHotCategoricalGen()
        {
            var gen = new Generator(4711);
            var categories = 7;
            var dist = OneHotCategorical(torch.rand(3, categories, dtype: ScalarType.Float64, generator: gen));
            {
                var sample = dist.sample();

                Assert.Equal(2, sample.ndim);
                Assert.Equal(3, sample.shape[0]);
                Assert.Equal(categories, sample.shape[1]);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(4, sample.ndim);
                Assert.Equal(new long[] { 2, 3, 3, 7 }, sample.shape);
            }
        }

        [Fact]
        public void TestCauchy()
        {
            var dist = Cauchy(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(new[] { 0.05, 0.10, 0.15 }));

            Assert.True(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { -2.7641, -2.8896, -2.7475 }).allclose(dist.log_prob(torch.ones(3))));
            Assert.True(torch.tensor(new[] { -0.46470802658, 0.22843915398, 0.633904262 }).allclose(dist.entropy()));

            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestCauchyGen()
        {
            var gen = new Generator(4711);
            var dist = Cauchy(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(new[] { 0.05, 0.10, 0.15 }), gen);
            Assert.True(torch.tensor(new[] { 0.5000, 0.2500, 0.1500 }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { -2.7641, -2.8896, -2.7475 }).allclose(dist.log_prob(torch.ones(3))));
            Assert.True(torch.tensor(new[] { -0.46470802658, 0.22843915398, 0.633904262 }).allclose(dist.entropy()));

            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestChi2()
        {
            var dist = Chi2(torch.tensor(new[] { 0.5, 0.25, 0.15 }));
            Assert.True(torch.tensor(new[] { 0.5, 0.25, 0.15 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 1, 0.5, 0.3 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { -1.9613093, -2.6060618, -3.1034275, }).allclose(dist.log_prob(torch.ones(3))));
            Assert.True(torch.tensor(new[] { -0.9394204, -4.502366, -9.439412 }).allclose(dist.entropy()));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestChi2Gen()
        {
            var gen = new Generator(4711);
            var dist = Chi2(torch.tensor(new[] { 0.5, 0.25, 0.15 }), gen);
            Assert.True(torch.tensor(new[] { 0.5, 0.25, 0.15 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 1, 0.5, 0.3 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { -1.9613093, -2.6060618, -3.1034275, }).allclose(dist.log_prob(torch.ones(3))));
            Assert.True(torch.tensor(new[] { -0.9394204, -4.502366, -9.439412 }).allclose(dist.entropy()));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestDirichlet()
        {
            var dist = Dirichlet(torch.tensor(new[] { 0.5, 0.25, 0.15 }));
            Assert.True(torch.tensor(new[] { 0.55555556, 0.27777778, 0.16666667, }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.12995452, 0.10558804, 0.073099415 }).allclose(dist.variance));
            Assert.True(torch.tensor(-4.9130).allclose(dist.entropy()));
            Assert.True(torch.tensor(-3.621825).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestDirichletGen()
        {
            var gen = new Generator(4711);
            var dist = Dirichlet(torch.tensor(new[] { 0.5, 0.25, 0.15 }), gen);
            Assert.True(torch.tensor(new[] { 0.55555556, 0.27777778, 0.16666667, }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.12995452, 0.10558804, 0.073099415 }).allclose(dist.variance));
            Assert.True(torch.tensor(-4.9130).allclose(dist.entropy()));
            Assert.True(torch.tensor(-3.621825).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestExponential()
        {
            var dist = Exponential(torch.tensor(new[] { 0.5, 0.25, 0.15 }));
            Assert.True(torch.tensor(new[] { 2, 4, 6.6666667, }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.0, 0.0, 0.0 }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { 0.25, 0.0625, 0.0225 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 1.6931472, 2.3862944, 2.89712 }).allclose(dist.entropy()));
            Assert.True(torch.tensor(new[] { -1.1931472, -1.6362944, -2.04712 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestExponentialGen()
        {
            var gen = new Generator(4711);
            var dist = Exponential(torch.tensor(new[] { 0.5, 0.25, 0.15 }), gen);
            Assert.True(torch.tensor(new[] { 2, 4, 6.6666667, }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 0.0, 0.0, 0.0 }).allclose(dist.mode));
            Assert.True(torch.tensor(new[] { 0.25, 0.0625, 0.0225 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { 1.6931472, 2.3862944, 2.89712 }).allclose(dist.entropy()));
            Assert.True(torch.tensor(new[] { -1.1931472, -1.6362944, -2.04712 }).allclose(dist.log_prob(torch.ones(3))));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestFisherSnedecor()
        {
            var dist = FisherSnedecor(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(2.0));
            Assert.True(torch.tensor(new[] { -2.0117974, -2.4718776, -2.8622819 }).allclose(dist.log_prob(torch.ones(3))));
            Assert.Throws<NotImplementedException>(() => dist.entropy());
            Assert.Throws<NotImplementedException>(() => dist.cdf(torch.ones(3)));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestFisherSnedecorGen()
        {
            var gen = new Generator(4711);
            var dist = FisherSnedecor(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(2.0), gen);
            Assert.True(torch.tensor(new[] { -2.0117974, -2.4718776, -2.8622819 }).allclose(dist.log_prob(torch.ones(3))));
            Assert.Throws<NotImplementedException>(() => dist.entropy());
            Assert.Throws<NotImplementedException>(() => dist.cdf(torch.ones(3)));
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestGamma()
        {
            var dist = Gamma(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(0.5f));
            Assert.True(torch.tensor(new[] { 1, 0.5, 0.3 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 2, 1, 0.6 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { -1.4189385, -1.9613093, -2.4317859 }).allclose(dist.log_prob(torch.ones(3))));
            Assert.True(torch.tensor(new[] { 0.7837571, -0.9394204, -3.296883, }).allclose(dist.entropy()));
            {
                var sample = dist.sample();
                var entropy = dist.entropy();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, 3 }, sample.shape);
            }
        }

        [Fact]
        public void TestGammaGen()
        {
            var gen = new Generator(4711);
            var dist = Gamma(torch.tensor(new[] { 0.5, 0.25, 0.15 }), torch.tensor(0.5f), gen);
            Assert.True(torch.tensor(new[] { 1, 0.5, 0.3 }).allclose(dist.mean));
            Assert.True(torch.tensor(new[] { 2, 1, 0.6 }).allclose(dist.variance));
            Assert.True(torch.tensor(new[] { -1.4189385, -1.9613093, -2.4317859 }).allclose(dist.log_prob(torch.ones(3))));
            Assert.True(torch.tensor(new[] { 0.7837571, -0.9394204, -3.296883, }).allclose(dist.entropy()));
            {
                var sample = dist.sample();
                var entropy = dist.entropy();

                Assert.Equal(new long[] { 3 }, sample.shape);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3 }, sample.shape);
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
        public void TestGeometricGen()
        {
            var gen = new Generator(4711);
            var dist = Geometric(torch.rand(3, 3), generator: gen);
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
                Assert.All<float>(sample.data<float>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, categories }, sample.shape);
                Assert.All<float>(sample.data<float>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, categories }, sample.shape);
                Assert.All<float>(sample.data<float>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
        }

        [Fact]
        public void TestMultinomialGen()
        {
            var gen = new Generator(4711);
            var categories = 17;
            var dist = Multinomial(100, torch.rand(3, categories), generator: gen);
            {
                var sample = dist.sample();

                Assert.Equal(new long[] { 3, categories }, sample.shape);
                Assert.All<float>(sample.data<float>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, categories }, sample.shape);
                Assert.All<float>(sample.data<float>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 3, categories }, sample.shape);
                Assert.All<float>(sample.data<float>().ToArray(), d => Assert.True(d >= 0 && d <= 100));
            }
        }

        [Fact]
        public void TestMultivariateNormal()
        {
            var bs = 2;
            var dist = torch.distributions.MultivariateNormal(torch.zeros(bs), precision_matrix: torch.eye(bs));
            {
                var sample = dist.sample(2, 2);

                Assert.Equal(new long[] { bs, 2, 2 }, sample.shape);
            }
            {
                torch.manual_seed(0);
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { bs, 3, 2 }, sample.shape);
            }
            {
                var sample = dist.expand(new long[] { 3, 3 }).sample(2, 3);

                Assert.Equal(new long[] { bs, 3, 3, 3, 2 }, sample.shape);
            }
        }

        [Fact]
        public void TestMultivariateNormal_1334()
        {
            var actionMean = torch.tensor(new double[]{0.2025, -0.0714, 0.1417});
            var covMat = torch.tensor(new double[,]
            {
                { 0.36, 0, 0 },
                { 0, 0.36, 0 },
                { 0, 0, 0.36 },
            });
            var dist = torch.distributions.MultivariateNormal(actionMean, covariance_matrix: covMat);
            torch.Tensor action = dist.sample();
            torch.Tensor actionLogProb = dist.log_prob(action);
        }

        [Fact]
        public void TestWeibull()
        {
            var dist = Weibull(torch.ones(3, 3), torch.ones(3, 3));
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
        public void TestWeibullGen()
        {
            var gen = new Generator(4711);
            var dist = Weibull(torch.ones(3, 3), torch.ones(3, 3), generator: gen);
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
    }
}
