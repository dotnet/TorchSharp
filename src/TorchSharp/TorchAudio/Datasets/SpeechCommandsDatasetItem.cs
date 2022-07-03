// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace TorchSharp
{
    public static partial class torchaudio
    {
        public static partial class datasets
        {
            public class SpeechCommandsDatasetItem
            {
                public torch.Tensor waveform;
                public int sample_rate;
                public string label;
                public string speaker_id;
                public int utterance_number;
            }
        }
    }
}