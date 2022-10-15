// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

namespace TorchSharp
{
    public static partial class torchaudio
    {
        public static partial class datasets
        {
            /// <summary>
            /// An item in SPEECHCOMMANDS dataset.
            /// </summary>
            public class SpeechCommandsDatasetItem
            {
                /// <summary>
                /// Samples of the audio clip
                /// </summary>
                public torch.Tensor waveform;

                /// <summary>
                /// Sampling rate of the audio clip
                /// </summary>
                public int sample_rate;

                /// <summary>
                /// Labels of the audio clip, 'bird', 'yes', ...
                /// </summary>
                public string label;

                /// <summary>
                /// Speaker ID
                /// </summary>
                public string speaker_id;

                /// <summary>
                /// Utterance number
                /// </summary>
                public int utterance_number;
            }
        }
    }
}