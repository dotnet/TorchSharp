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
        public class Bernoulli : torch.distributions.Distribution
        {

            public override Tensor mean => probs;

            public override Tensor variance => probs * (1 - probs);

            public Bernoulli(Tensor p = null, Tensor l = null) 
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
                return torch.bernoulli(probs.expand(shape));
            }

            public override Tensor log_prob(Tensor value)
            {
                var logitsValue = torch.broadcast_tensors(logits, value);
                return -torch.nn.functional.binary_cross_entropy_with_logits(logitsValue[0], logitsValue[1], reduction: nn.Reduction.None);
            }

            public override Tensor entropy()
            {
                return torch.nn.functional.binary_cross_entropy_with_logits(logits, probs, reduction: nn.Reduction.None);
            }

            public override distributions.Distribution expand(long[] batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Bernoulli))
                    throw new ArgumentException("expand(): 'instance' must be a Bernoulli distribution");

                var p = _probs?.expand(batch_shape);
                var l = _logits?.expand(batch_shape);

                var newDistribution = ((instance == null) ? new Bernoulli(p, l) : instance) as Bernoulli;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
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
            /// Creates a Bernoulli distribution parameterized by `probs` or `logits` (but not both).
            /// </summary>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <returns></returns>
            public static Bernoulli Bernoulli(Tensor probs = null, Tensor logits = null)
            {
                return new Bernoulli(probs, logits);
            }
        }
    }
}
