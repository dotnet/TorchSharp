using System;
using System.Collections.Generic;
using System.Text;
using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    internal class RandomChoice : IDisposable, ITransform
    {
        public RandomChoice(ITransform[] transforms)
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

        public Tensor forward(Tensor input)
        {
            using (var chance = Int32Tensor.randint(transforms.Length, new long[] { 1 }))
                return transforms[chance.DataItem<int>()].forward(input);
        }

        private ITransform[] transforms;
    }

    public static partial class transforms
    {
        /// <summary>
        /// Apply a single transformation randomly picked from a list. 
        /// </summary>
        /// <param name="transforms">A list of transforms to apply.</param>
        static public ITransform RandomChoice(params ITransform[] transforms)
        {
            return new RandomChoice(transforms);
        }
    }
}
