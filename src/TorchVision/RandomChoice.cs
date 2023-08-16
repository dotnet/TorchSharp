// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class RandomChoice : IDisposable, ITransform
        {
            public RandomChoice(Generator generator, ITransform[] transforms)
            {
                this.transforms = transforms;
                this.generator = generator;
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
                var chance = torch.randint_int(transforms.Length, generator);
                return transforms[chance].call(input);
            }

            private ITransform[] transforms;
            private Generator generator;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Apply a single transformation randomly picked from a list. 
            /// </summary>
            /// <param name="transforms">A list of transforms to apply.</param>
            static public ITransform RandomChoice(params ITransform[] transforms)
            {
                return new RandomChoice(null, transforms);
            }

            /// <summary>
            /// Apply a single transformation randomly picked from a list. 
            /// </summary>
            /// <param name="generator">A random number generator instance.</param>
            /// <param name="transforms">A list of transforms to apply.</param>
            static public ITransform RandomChoice(Generator generator, params ITransform[] transforms)
            {
                return new RandomChoice(generator, transforms);
            }
        }
    }
}
