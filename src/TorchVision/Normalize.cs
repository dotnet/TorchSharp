// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;


namespace TorchSharp
{
    public static partial class torchvision
    {
        internal class Normalize : ITransform, IDisposable
        {
            internal Normalize(double[] means, double[] stdevs, ScalarType dtype = ScalarType.Float32, torch.Device device = null)
            {
                if (means.Length != stdevs.Length) throw new ArgumentException("means and stdevs must be the same length in call to Normalize");

                this.means = means.ToTensor(new long[] { 1, means.Length, 1, 1 });     // Assumes NxCxHxW
                this.stdevs = stdevs.ToTensor(new long[] { 1, stdevs.Length, 1, 1 });  // Assumes NxCxHxW

                if (dtype != ScalarType.Float64) {
                    this.means = this.means.to_type(dtype);
                    this.stdevs = this.stdevs.to_type(dtype);
                }

                if (device != null && device.type != DeviceType.CPU) {
                    this.means = this.means.to(device);
                    this.stdevs = this.stdevs.to(device);
                }
            }

            public Tensor call(Tensor input)
            {
                if (means.size(1) != input.size(1)) throw new ArgumentException("The number of channels is not equal to the number of means and standard deviations");
                return (input - means) / stdevs;
            }

            private Tensor means;
            private Tensor stdevs;
            bool disposedValue;

            protected virtual void Dispose(bool disposing)
            {
                if (!disposedValue) {
                    means.Dispose();
                    stdevs.Dispose();
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
            /// <param name="dtype">Bool to make this operation inplace.</param>
            /// <param name="device">The device to place the output tensor on.</param>
            /// <returns></returns>
            static public ITransform Normalize(double[] means, double[] stdevs, ScalarType dtype = ScalarType.Float32, torch.Device device = null)
            {
                return new Normalize(means, stdevs, dtype, device);
            }
        }
    }
}
