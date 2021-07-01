using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp.Tensor;

namespace TorchSharp.torchvision
{
    internal class ComposedTransforms : IDisposable, ITransform
    {
        public ComposedTransforms(ITransform[] transforms)
        {
            this.transforms = transforms;
        }

        public void Dispose()
        {
            foreach (var t in transforms) {
                if (t is IDisposable) {
                    ((IDisposable)t).Dispose();
                }
            }
        }

        public TorchTensor forward(TorchTensor input)
        {
            foreach (var t in transforms) {
                input = t.forward(input);
            }
            return input;
        }

        private ITransform[] transforms;
    }

    public static partial class transforms
    {
        static public ITransform Compose(params ITransform[] transforms)
        {
            return new ComposedTransforms(transforms);
        }
    }
}
