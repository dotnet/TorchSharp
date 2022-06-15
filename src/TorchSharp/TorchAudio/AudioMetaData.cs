// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
namespace TorchSharp
{
    public static partial class torchaudio
    {
        /// <summary>
        /// Meta-data of audio
        /// </summary>
        public struct AudioMetaData
        {
            /// <summary>
            /// Sample rate
            /// </summary>
            public int sample_rate;

            /// <summary>
            /// The number of frames
            /// </summary>
            public int num_frames;

            /// <summary>
            /// The number of channels
            /// </summary>
            public int num_channels;

            /// <summary>
            /// The number of bits per sample.
            /// </summary>
            public int bits_per_sample;

            /// <summary>
            /// Audio encoding
            /// </summary>
            public AudioEncoding encoding;
        }
    }
}