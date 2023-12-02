// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class distributions
        {
            public abstract class Distribution : IDisposable
            {
                public Distribution(torch.Generator generator, long[] batch_shape = null, long[] event_shape = null)
                {
                    this.generator = generator;
                    _init(batch_shape != null ? batch_shape : Size.Empty,
                          event_shape != null ? event_shape : Size.Empty);
                }

                public Distribution(torch.Generator generator, Size batch_shape, Size? event_shape = null)
                {
                    this.generator = generator;
                    _init(batch_shape,
                          event_shape != null ? event_shape : Size.Empty);
                }

                protected void _init(Size? batch_shape = null, Size? event_shape = null)
                {
                    this.batch_shape = batch_shape != null ? batch_shape.Value : Size.Empty;
                    this.event_shape = event_shape != null ? event_shape.Value : Size.Empty;
                }

                /// <summary>
                /// The shape over which parameters are batched.
                /// </summary>
                public Size batch_shape { get; protected set; }

                /// <summary>
                /// The shape of a single sample (without batching).
                /// </summary>
                public Size event_shape { get; protected set; }

                /// <summary>
                /// The mean of the distribution.
                /// </summary>
                public abstract Tensor mean { get; }

                /// <summary>
                /// The mode of the distribution.
                /// </summary>
                public virtual Tensor mode { get { return new Tensor(IntPtr.Zero); } } 

                /// <summary>
                /// The variance of the distribution
                /// </summary>
                public abstract Tensor variance { get; }

                /// <summary>
                /// The standard deviation of the distribution
                /// </summary>
                public virtual Tensor stddev => variance.sqrt();

                /// <summary>
                /// Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched.
                /// </summary>
                /// <param name="sample_shape">A list of dimension sizes</param>
                /// <returns>A tensor containing the sample.</returns>
                public virtual Tensor sample(params long[] sample_shape)
                {
                    return rsample(sample_shape);
                }

                /// <summary>
                /// Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched.
                /// </summary>
                /// <param name="sample_shape">A list of dimension sizes</param>
                /// <returns>A tensor containing the sample.</returns>
                public Tensor sample(Size sample_shape) => sample(sample_shape.Shape);

                /// <summary>
                ///  Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples
                ///  if the distribution parameters are batched.
                /// </summary>
                /// <param name="sample_shape">The sample shape.</param>
                /// <returns>A tensor containing the sample.</returns>
                public abstract Tensor rsample(params long[] sample_shape);

                /// <summary>
                ///  Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples
                ///  if the distribution parameters are batched.
                /// </summary>
                /// <param name="sample_shape">The sample shape.</param>
                /// <returns>A tensor containing the sample.</returns>
                public Tensor rsample(Size sample_shape) => rsample(sample_shape.Shape);

                /// <summary>
                /// Returns the log of the probability density/mass function evaluated at `value`.
                /// </summary>
                /// <param name="value"></param>
                /// <returns></returns>
                public abstract Tensor log_prob(Tensor value);

                /// <summary>
                /// Returns entropy of distribution, batched over batch_shape.
                /// </summary>
                /// <returns></returns>
                public abstract Tensor entropy();

                /// <summary>
                /// Returns a new distribution instance (or populates an existing instance provided by a derived class) with batch dimensions expanded to
                /// `batch_shape`. This method calls `torch.Tensor.expand()` on the distribution's parameters. As such, this does not allocate new
                /// memory for the expanded distribution instance.
                /// </summary>
                /// <param name="batch_shape">Tthe desired expanded size.</param>
                /// <param name="instance">new instance provided by subclasses that need to override `.expand`.</param>
                /// <returns></returns>
                public abstract Distribution expand(Size batch_shape, Distribution instance = null);

                /// <summary>
                /// Returns the cumulative density/mass function evaluated at `value`.
                /// </summary>
                /// <param name="value"></param>
                /// <returns></returns>
                public virtual Tensor cdf(Tensor value)
                {
                    throw new NotImplementedException("Distribution.cdf()");
                }

                /// <summary>
                /// Returns the inverse cumulative density/mass function evaluated at `value`.
                /// </summary>
                /// <param name="value"></param>
                /// <returns></returns>
                public virtual Tensor icdf(Tensor value)
                {
                    throw new NotImplementedException("Distribution.icdf()");
                }

                /// <summary>
                /// Returns tensor containing all values supported by a discrete distribution. The result will enumerate over dimension 0, so the shape
                /// of the result will be `(cardinality,) + batch_shape + event_shape` (where `event_shape = ()` for univariate distributions).
                ///
                /// Note that this enumerates over all batched tensors in lock-step `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
                /// along dim 0, but with the remaining batch dimensions being singleton dimensions, `[[0], [1], ..`
                /// </summary>
                /// <param name="expand">Whether to expand the support over the batch dims to match the distribution's `batch_shape`.</param>
                /// <returns></returns>
                public virtual Tensor enumerate_support(bool expand = true)
                {
                    throw new NotImplementedException();
                }

                /// <summary>
                /// Returns perplexity of distribution, batched over batch_shape.
                /// </summary>
                /// <returns></returns>
                public virtual Tensor perplexity() => torch.exp(entropy());

                protected long[] ExtendedShape(params long[] sample_shape)
                {
                    if (batch_shape.Length == 0 && event_shape.Length == 0)
                        return sample_shape;

                    var result = new List<long>();
                    if (sample_shape.Length > 0) result.AddRange(sample_shape);
                    if (batch_shape.Length > 0) result.AddRange(batch_shape);
                    if (event_shape.Length > 0) result.AddRange(event_shape);

                    return result.ToArray();
                }

                protected Tensor LogitsToProbs(Tensor logits, bool isBinary = false)
                {
                    return (isBinary) ? torch.sigmoid(logits) : torch.nn.functional.softmax(logits, dim: -1);
                }

                protected Tensor ProbsToLogits(Tensor probs, bool isBinary = false)
                {
                    probs = ClampProbs(probs);
                    return (isBinary) ? (torch.log(probs) - torch.log1p(-probs)) : torch.log(probs);
                }

                protected Tensor ClampProbs(Tensor probs)
                {
                    var eps = torch.finfo(probs.dtype).eps;
                    return probs.clamp(eps, 1 - eps);
                }

                protected Tensor ClampByZero(Tensor x) => (x.clamp_min(0) + x - x.clamp_max(0)) / 2;

                protected torch.Generator generator;
                bool disposedValue;
                protected const double euler_constant = 0.57721566490153286060; // Euler Mascheroni Constant

                protected virtual void Dispose(bool disposing)
                {
                    if (!disposedValue) {
                        if (disposing) {
                            generator?.Dispose();
                        }
                        disposedValue = true;
                    }
                }

                ~Distribution()
                {
                    // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
                    Dispose(disposing: false);
                }

                public void Dispose()
                {
                    // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
                    Dispose(disposing: true);
                    GC.SuppressFinalize(this);
                }
            }
        }
    }
}
