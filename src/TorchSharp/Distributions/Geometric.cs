using System;
using System.Collections.Generic;
using System.Text;

using TorchSharp;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public class Geometric : torch.distributions.Distribution
        {

            public override Tensor mean => 1 / (probs - 1);

            public override Tensor variance => (1.0f / probs - 1.0f) / probs;

            public Geometric(Tensor p = null, Tensor l = null) 
            {
                this.batch_shape = p is null ? l.size() : p.size();
                this._probs = p;
                this._logits = l;
            }

            public Tensor probs {
                get {
                    return _probs ?? LogitsToProbs(_logits, true);
                }
            }
            public Tensor logits {
                get {
                    return _logits ?? ProbsToLogits(_probs);
                }
            }

            private Tensor _probs;
            private Tensor _logits;

            public override Tensor rsample(params long[] sample_shape)
            {
                var shape = ExtendedShape(sample_shape);
                var tiny = torch.finfo(probs.dtype).tiny;
                using (torch.no_grad()) {
                    var u = probs.new_empty(shape).uniform_(tiny, 1);
                    return (u.log() / (-probs).log1p()).floor();
                }
            }

            public override Tensor log_prob(Tensor value)
            {
                var bcast = torch.broadcast_tensors(value, probs);
                value = bcast[0];
                var p = bcast[1].clone();
                p[(p == 1) & (value == 0)] = torch.tensor(0);
                return value * (-p).log1p() + probs.log();
            }

            public override Tensor entropy()
            {
                return torch.nn.functional.binary_cross_entropy_with_logits(logits, probs, reduction: nn.Reduction.None);
            }

            public override distributions.Distribution expand(long[] batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Geometric))
                    throw new ArgumentException("expand(): 'instance' must be a Geometric distribution");

                var newDistribution = ((instance == null) ?
                    new Geometric(p: _probs?.expand(batch_shape), l: logits?.expand(batch_shape)) :
                    instance) as Geometric;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution._probs = _probs?.expand(batch_shape);
                    newDistribution._logits = _logits?.expand(batch_shape);
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
            /// Creates a Geometric distribution parameterized by probs,
            /// where probs is the probability of success of Bernoulli trials.
            ///
            /// It represents the probability that in k+1 Bernoulli trials, the
            /// first k trials failed, before seeing a success.
            /// </summary>
            /// <param name="probs">The probability of sampling '1'. Must be in range (0, 1]</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <returns></returns>
            public static Geometric Geometric(Tensor probs = null, Tensor logits = null)
            {
                return new Geometric(probs, logits);
            }
        }
    }
}
