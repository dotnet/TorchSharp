// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
namespace TorchSharp
{
    public static partial class torchaudio
    {
        public struct AudioMetaData
        {
            public int sample_rate;
            public int num_frames;
            public int num_channels;
            public int bits_per_sample;
            public AudioEncoding encoding;
        }
    }
}