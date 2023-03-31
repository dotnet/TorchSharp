// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
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

            public Tensor call(Tensor input)
            {
                using (var chance = torch.randint(transforms.Length, new long[] { 1 }, ScalarType.Int32))
                    return transforms[chance.item<int>()].call(input);
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
}
