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
        public void TestUniform()
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
        public void TestUniformGen()
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
        public void TestNormalGen()
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
        public void TestHalfNormal()
        {
            var dist = HalfNormal(torch.tensor(3.5));
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
        public void TestHalfNormalGen()
        {
            var dist = HalfNormal(torch.tensor(3.5), new Generator(4711));
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
        public void TestHalfCauchy()
        {
            var dist = HalfCauchy(torch.tensor(3.5));
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
        public void TestHalfCauchyGen()
        {
            var dist = HalfCauchy(torch.tensor(3.5), new Generator(4711));
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
        public void TestPareto()
        {
            var dist = Pareto(torch.tensor(1.5), torch.tensor(1.0));
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
        public void TestParetoGen()
        {
            var dist = Pareto(torch.tensor(1.5), torch.tensor(1.0), new Generator(4711));
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
        public void TestLogNormal()
        {
            var dist = LogNormal(torch.tensor(0.25), torch.tensor(3.5));
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
        public void TestLogNormalGen()
        {
            var dist = LogNormal(torch.tensor(0.25), torch.tensor(3.5), new Generator(4711));
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
        public void TestGumbel()
        {
            var dist = Gumbel(torch.tensor(0.0), torch.tensor(3.5));
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
        public void TestGumbelGen()
        {
            var dist = Gumbel(torch.tensor(0.0), torch.tensor(3.5), new Generator(4711));
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
        public void TestLaplace()
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
        public void TestLaplaceGen()
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
        public void TestPoissonGen()
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
        public void TestBernoulli()
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
        public void TestBernoulliGen()
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
        public void TestBinomialGen()
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
            var dist = OneHotCategorical(torch.rand(3, categories, dtype: ScalarType.Float64, generator:gen));
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
        public void TestCauchyGen()
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
        public void TestChi2Gen()
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
        public void TestDirichletGen()
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
        public void TestExponentialGen()
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
        public void TestFisherSnedecorGen()
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
        public void TestGammaGen()
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
        public void TestWeibull()
        {
            var dist = Weibull(torch.ones(3,3), torch.ones(3, 3));
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
