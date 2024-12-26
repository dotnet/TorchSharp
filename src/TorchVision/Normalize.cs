// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Normalize : ITransform, IDisposable
        {
            internal Normalize(double[] means, double[] stdevs,bool inplace = false)
            {
                if (means is null) throw new ArgumentNullException(nameof(means));
                if (stdevs is null) throw new ArgumentNullException(nameof(stdevs));
                if (means.Length != stdevs.Length)
                    throw new ArgumentException($"{nameof(means)} and {nameof(stdevs)} must be the same length in call to Normalize");
                if (means.Length != 1 && means.Length != 3)
                    throw new ArgumentException($"Since they correspond to the number of channels in an image, {nameof(means)} and {nameof(stdevs)} must both be either 1 or 3 long");
                this.means = means;
                this.stdevs = stdevs;
                this.inplace = inplace;

            }

            public Tensor call(Tensor input)
            {
                return transforms.functional.normalize(input, means, stdevs, inplace);
            }

            private readonly double[] means;
            private readonly double[] stdevs;
            private readonly bool inplace;
            bool disposedValue;

            protected virtual void Dispose(bool disposing)
            {
                if (!disposedValue) {
                    disposedValue = true;
                }
            }

            ~Normalize()
            {
                // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
                Dispose(disposing: false);
            }

            public void Dispose()
            {
                // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
                Dispose(disposing: true);
                GC.SuppressFinalize(this);
            }
        }

        public static partial class transforms
        {
            /// <summary>
            /// Normalize a float tensor image with mean and standard deviation.
            /// </summary>
            /// <param name="means">Sequence of means for each channel.</param>
            /// <param name="stdevs">Sequence of standard deviations for each channel.</param>
            /// <param name="inplace">Bool to make this operation inplace.</param>
            /// <returns></returns>
            static public ITransform Normalize(double[] means, double[] stdevs, bool inplace = false)
            {
                return new Normalize(means, stdevs, inplace);
            }
        }
    }
}
