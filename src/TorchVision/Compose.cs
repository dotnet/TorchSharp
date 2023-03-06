// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
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

            public Tensor call(Tensor input)
            {
                foreach (var t in transforms) {
                    input = t.call(input);
                }
                return input;
            }

            private ITransform[] transforms;
        }

        public static partial class transforms
        {
            /// <summary>
            /// Composes several transforms together.
            /// </summary>
            /// <param name="transforms">A list of transforms to compose serially.</param>
            /// <returns></returns>
            static public ITransform Compose(params ITransform[] transforms)
            {
                return new ComposedTransforms(transforms);
            }
        }
    }
}
