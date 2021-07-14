using System;
using System.Collections.Generic;
using System.Linq;

using TorchSharp;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public class Categorical : torch.distributions.Distribution
        {

            public override Tensor mean => torch.full(ExtendedShape(), double.NaN, dtype: probs.dtype, device: probs.device);

            public override Tensor variance => torch.full(ExtendedShape(), double.NaN, dtype: probs.dtype, device: probs.device);

            public Categorical(Tensor p = null, Tensor l = null) 
            {
                var param = p is null ? l : p;

                this._probs = (p is not null) ? p / p.sum(-1, keepdim: true) : null;
                this._logits = (l is not null) ? (l - l.logsumexp(-1, keepdim:true)) : null;
                this.num_events = param.size(-1);
                this.batch_shape = param.ndim > 1 ? param.shape.Take(param.shape.Length-1).ToArray() : new long[0];
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
            private long num_events;

            public override Tensor rsample(params long[] sample_shape)
            {
                var probs_2d = probs.reshape(-1, num_events);
                var samples_2d = torch.multinomial(probs_2d, sample_shape.Aggregate<long, long>(1, (x, y) => x * y), true).T;
                return samples_2d.reshape(ExtendedShape(sample_shape));
            }

            public override Tensor log_prob(Tensor value)
            {
                value = value.@long().unsqueeze(-1);
                var valLogPmf = torch.broadcast_tensors(value, logits);
                value = valLogPmf[0][TensorIndex.Ellipsis, TensorIndex.Slice(null, 1)];
                return valLogPmf[1].gather(-1, value).squeeze(-1);
            }

            public override Tensor entropy()
            {
                var min_real = torch.finfo(logits.dtype).min;
                var logs = torch.clamp(logits, min: min_real);
                var p_log_p = logs * probs;
                return -p_log_p.sum(-1);
            }

            public override distributions.Distribution expand(long[] batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Categorical))
                    throw new ArgumentException("expand(): 'instance' must be a Categorical distribution");

                var param_shape = new List<long>();
                param_shape.AddRange(batch_shape);
                param_shape.Add(num_events);

                var shape = param_shape.ToArray();

                var p = _probs?.expand(shape);
                var l = _logits?.expand(shape);

                var newDistribution = ((instance == null) ? new Categorical(p, l) : instance) as Categorical;

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
            /// <returns></returns>
            public static Categorical Categorical(Tensor probs = null, Tensor logits = null)
            {
                return new Categorical(probs, logits);
            }
        }
    }
}
