// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
namespace TorchSharp
{
    public static partial class torchaudio
    {
        /// <summary>
        /// Resampling method
        /// </summary>
        public enum ResamplingMethod
        {
            sinc_interpolation,
            kaiser_window
        }
    }
}