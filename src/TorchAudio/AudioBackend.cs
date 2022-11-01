// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

namespace TorchSharp
{
    public static partial class torchaudio
    {
        public static partial class backend
        {
            /// <summary>
            /// Audio I/O backend
            /// </summary>
            public abstract class AudioBackend
            {
                public abstract (torch.Tensor, int) load(
                    string filepath,
                    long frame_offset = 0,
                    long num_frames = -1,
                    bool normalize = true,
                    bool channels_first = true,
                    AudioFormat? format = null);

                public abstract void save(
                    string filepath,
                    torch.Tensor src,
                    int sample_rate,
                    bool channels_first = true,
                    float? compression = null,
                    AudioFormat? format = null,
                    AudioEncoding? encoding = null,
                    int? bits_per_sample = null);

                public abstract AudioMetaData info(
                    string filepath,
                    AudioFormat? format = null);
            }
        }
    }
}
