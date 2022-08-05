// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;
using static TorchSharp.torch.distributions;

using Xunit;

#nullable enable

namespace TorchSharp
{
#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
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
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d < 3.5));
            }
            {
                var sample = dist.expand(new long[] { 3, 4 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 4 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d < 3.5));
            }
        }

        [Fact]
        public void TestUniform2()
        {
            var gen = new Generator(4711);

            var dist = Uniform(torch.tensor(0.0), torch.tensor(3.5), gen);
            {
                var sample = dist.sample();

                Assert.Empty(sample.shape);
                Assert.True(sample.ToDouble() >= 0.0 && sample.ToDouble() < 3.5);
            }
            {
                var sample = dist.sample(2, 3);

                Assert.Equal(new long[] { 2, 3 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d < 3.5));
            }
            {
                var sample = dist.expand(new long[] { 3, 4 }).sample(2, 3);

                Assert.Equal(new long[] { 2, 3, 3, 4 }, sample.shape);
                Assert.All<double>(sample.data<double>().ToArray(), d => Assert.True(d >= 0 && d < 3.5));
            }
        }

        [Fact]
        public void TestNormal1()
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
        public void TestNormal2()
        {
            var gen = new Generator(4711);

            var dist = Normal(torch.tensor(0.0), torch.tensor(3.5), gen);
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
        public void TestLaplace1()
        {
            var dist = Laplace(torch.tensor(0.0), torch.tensor(3.5));
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
        public void TestLaplace2()
        {
            var gen = new Generator(4711);

            var dist = Laplace(torch.tensor(0.0), torch.tensor(3.5), gen);
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
        public void TestPoisson1()
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
        public void TestPoisson2()
        {
            var gen = new Generator(4711);
            var dist = Poisson(torch.tensor(0.5), gen);
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
        public void TestBernoulli1()
        {
            var dist = Bernoulli(torch.rand(3, dtype: ScalarType.Float64));
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
        public void TestBernoulli2()
        {
            var gen = new Generator(4711);
            var dist = Bernoulli(torch.rand(3, dtype: ScalarType.Float64), generator: gen);
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
        public void TestLogitRelaxedBernoulli1()
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
        public void TestLogitRelaxedBernoulli2()
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
        public void TestBeta1()
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
        public void TestBeta2()
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
        public void TestBinomial1()
        {
            var dist = Binomial(torch.tensor(100), torch.rand(3, dtype: ScalarType.Float64));
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
        public void TestBinomial2()
        {
            var gen = new Generator(4711);
            var dist = Binomial(torch.tensor(100), torch.rand(3, dtype: ScalarType.Float64), generator: gen);
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
        public void TestCategorical1()
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
        public void TestCategorical2()
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
        public void TestCauchy1()
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
        public void TestCauchy2()
        {
            var gen = new Generator(4711);
            var dist = Cauchy(torch.rand(3, 3), torch.tensor(1.0f), gen);
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
        public void TestChi21()
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
        public void TestChi22()
        {
            var gen = new Generator(4711);
            var dist = Chi2(torch.rand(3, 3), gen);
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
        public void TestDirichlet1()
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
        public void TestDirichlet2()
        {
            var gen = new Generator(4711);
            var dist = Dirichlet(torch.rand(3, 3) * 0.5f, gen);
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
        public void TestExponential1()
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
        public void TestExponential2()
        {
            var gen = new Generator(4711);
            var dist = Exponential(torch.rand(3, 3) * 0.5f, gen);
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
        public void TestFisherSnedecor1()
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
        public void TestFisherSnedecor2()
        {
            var gen = new Generator(4711);
            var dist = FisherSnedecor(torch.rand(3, 3) * 1.5f, torch.tensor(2.0f), gen);
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
        public void TestGamma1()
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
        public void TestGamma2()
        {
            var gen = new Generator(4711);
            var dist = Gamma(torch.rand(3, 3), torch.tensor(1.0f), gen);
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
        public void TestGeometric1()
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
        public void TestGeometric2()
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
        public void TestMultinomial1()
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
        public void TestMultinomial2()
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
    }
}
