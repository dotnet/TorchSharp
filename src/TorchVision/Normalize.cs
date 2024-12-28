// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Normalize : ITransform
        {
            internal Normalize(double[] means, double[] stdevs,bool inplace = false)
            {
                if (means is null) throw new ArgumentNullException(nameof(means));
                if (stdevs is null) throw new ArgumentNullException(nameof(stdevs));
                if (means.Length != stdevs.Length)
                    throw new ArgumentException($"{nameof(means)} and {nameof(stdevs)} must be the same length in call to Normalize");
                this.means = means;
                this.stdevs = stdevs;
                this.inplace = inplace;

            }

            public Tensor call(Tensor input)
            {
                var expectedChannels = (input.ndim == 4) ? input.size(1) : input.size(0);
                if (expectedChannels != means.Length)
                    throw new ArgumentException("The number of channels is not equal to the number of means and standard deviations");
                return transforms.functional.normalize(input, means, stdevs, inplace);
            }

            private readonly double[] means;
            private readonly double[] stdevs;
            private readonly bool inplace;
         
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
