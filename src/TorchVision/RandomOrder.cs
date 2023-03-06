// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class RandomOrder : IDisposable, ITransform
        {
            public RandomOrder(ITransform[] transforms)
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
                var rng = new Random();
                foreach (var t in transforms.OrderBy(t => rng.NextDouble())) {
                    input = t.call(input);
                }
                return input;
            }

            private IList<ITransform> transforms;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Apply a list of transformations in a random order.
            /// </summary>
            /// <param name="transforms">A list of transforms to apply.</param>
            /// <remarks>
            /// This transform uses the .NET Random API, not the Torch RNG.
            /// Each invocation of 'forward()' will randomize the order in which transforms
            /// are applied.
            /// </remarks>
            static public ITransform RandomOrder(params ITransform[] transforms)
            {
                return new RandomOrder(transforms);
            }
        }
    }
}