using System;
using System.Collections.Generic;
using System.Linq;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class distributions
        {
            public abstract class ExponentialFamily : Distribution
            {

                protected abstract IList<Tensor> NaturalParams { get; }

                protected virtual Tensor LogNormalizer(params Tensor[] parameters)
                {
                    throw new NotImplementedException();
                }

                protected abstract Tensor MeanCarrierMeasure { get; }

                public override Tensor entropy()
                {
                    // Method to compute the entropy using Bregman divergence of the log normalizer.

                    var result = -MeanCarrierMeasure;
                    var nparams = NaturalParams.Select(p => p.detach().requires_grad_()).ToArray();
                    var lg_normal = LogNormalizer(nparams);
                    var gradients = torch.autograd.grad(new Tensor[] { lg_normal.sum() }, nparams, create_graph: true);
                    result += lg_normal;
                    for (int i = 0; i < gradients.Count; i++) {
                        var np = nparams[i];
                        var g = gradients[i];
                        result -= np * g;
                    }
                    return result;
                }
            }
        }
    }
}
