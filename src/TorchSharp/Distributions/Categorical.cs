// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public class Categorical : torch.distributions.Distribution
        {
            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean => torch.full(ExtendedShape(), double.NaN, dtype: probs.dtype, device: probs.device);

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance => torch.full(ExtendedShape(), double.NaN, dtype: probs.dtype, device: probs.device);

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="probs">Event probabilities</param>
            /// <param name="logits">Even log-odds</param>
            /// <param name="generator">An optional random number generator object.</param>
            public Categorical(Tensor probs = null, Tensor logits = null, torch.Generator generator = null) : base(generator)
            {
                var param = probs is null ? logits : probs;

                this._probs = (probs is not null) ? (probs / probs.sum(-1, keepdim: true)).DetachFromDisposeScope() : null;
                this._logits = (logits is not null) ? (logits - logits.logsumexp(-1, keepdim:true)).DetachFromDisposeScope() : null;
                this.num_events = param.size(-1);
                this.batch_shape = param.ndim > 1 ? param.shape.Take(param.shape.Length-1).ToArray() : new long[0];
            }

            /// <summary>
            /// Event probabilities
            /// </summary>
            public Tensor probs {
                get {
                    return _probs ?? LogitsToProbs(_logits);
                }
            }

            /// <summary>
            /// Event log-odds
            /// </summary>
            public Tensor logits {
                get {
                    return _logits ?? ProbsToLogits(_probs);
                }
            }

            /// <summary>
            ///  The shape of the input parameter.
            /// </summary>
            public long[] param_shape {
                get {
                    return _probs is null ? _logits.shape : _probs.shape;
                }
            }

            internal Tensor _probs;
            internal Tensor _logits;
            internal long num_events;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    _probs?.Dispose();
                    _logits?.Dispose();
                }
                base.Dispose(disposing);
            }

            /// <summary>
            ///  Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples
            ///  if the distribution parameters are batched.
            /// </summary>
            /// <param name="sample_shape">The sample shape.</param>
            public override Tensor rsample(params long[] sample_shape)
            {
                var probs_2d = probs.reshape(-1, num_events);
                var samples_2d = torch.multinomial(probs_2d, sample_shape.Aggregate<long, long>(1, (x, y) => x * y), true, generator).T;
                return samples_2d.reshape(ExtendedShape(sample_shape));
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value)
            {
                value = value.@long().unsqueeze(-1);
                var valLogPmf = torch.broadcast_tensors(value, logits);
                value = valLogPmf[0][TensorIndex.Ellipsis, TensorIndex.Slice(null, 1)];
                return valLogPmf[1].gather(-1, value).squeeze(-1);
            }

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            public override Tensor entropy()
            {
                var min_real = torch.finfo(logits.dtype).min;
                var logs = torch.clamp(logits, min: min_real);
                var p_log_p = logs * probs;
                return -p_log_p.sum(-1);
            }

            /// <summary>
            /// Returns a new distribution instance (or populates an existing instance provided by a derived class) with batch dimensions expanded to
            /// `batch_shape`. This method calls `torch.Tensor.expand()` on the distribution's parameters. As such, this does not allocate new
            /// memory for the expanded distribution instance.
            /// </summary>
            /// <param name="batch_shape">Tthe desired expanded size.</param>
            /// <param name="instance">new instance provided by subclasses that need to override `.expand`.</param>
            /// <returns></returns>
            public override distributions.Distribution expand(Size batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Categorical))
                    throw new ArgumentException("expand(): 'instance' must be a Categorical distribution");

                var param_shape = new List<long>();
                param_shape.AddRange(batch_shape);
                param_shape.Add(num_events);

                var shape = param_shape.ToArray();

                var p = _probs?.expand(shape);
                var l = _logits?.expand(shape);

                var newDistribution = ((instance == null) ? new Categorical(p, l, generator) : instance) as Categorical;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution.num_events = this.num_events;
                    newDistribution._probs = p;
                    newDistribution._logits = l;
                }
                return newDistribution;
            }
        }

    }
    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Creates a Categorical distribution parameterized by `probs` or `logits` (but not both).
            ///
            /// Samples are integers from [0, K-1]` where `K` is probs.size(-1).
            /// 
            /// If `probs` is 1-dimensional with length- `K`, each element is the relative probability
            /// of sampling the class at that index.
            /// 
            /// If `probs` is N-dimensional, the first N-1 dimensions are treated as a batch of
            /// relative probability vectors.
            /// </summary>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Categorical Categorical(Tensor probs = null, Tensor logits = null, torch.Generator generator = null)
            {
                return new Categorical(probs, logits);
            }

            /// <summary>
            /// Creates an equal-probability categorical distribution parameterized by the number of categories.
            ///
            /// </summary>
            /// <param name="categories">The number of categories.</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Categorical Categorical(int categories, torch.Generator generator = null)
            {
                var probs = torch.tensor(1.0 / categories).expand(categories);
                return new Categorical(probs, null, generator);
            }
        }
    }
}
