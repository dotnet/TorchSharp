// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Text;

namespace TorchSharp
{
    public static partial class torchaudio
    {
        public static partial class backend
        {
            public class NoBackend : IAudioBackend
            {
                public (torch.Tensor, int) load(
                    string filepath,
                    int frame_offset = 0,
                    int num_frames = -1,
                    bool normalize = true,
                    bool channels_first = true,
                    AudioFormat? format = null)
                {
                    throw new InvalidOperationException("No audio I/O backend is available.");
                }

                public void save(
                    string filepath,
                    torch.Tensor src,
                    int sample_rate,
                    bool channels_first = true,
                    float? compression = null,
                    AudioFormat? format = null,
                    AudioEncoding? encoding = null,
                    int? bits_per_sample = null)
                {
                    throw new InvalidOperationException("No audio I/O backend is available.");
                }

                public AudioMetaData info(
                    string filepath,
                    AudioFormat? format = null)
                {
                    throw new InvalidOperationException("No audio I/O backend is available.");
                }
            }
        }
    }
}
