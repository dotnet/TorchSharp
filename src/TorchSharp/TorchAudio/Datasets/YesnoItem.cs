// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;

namespace TorchSharp
{
    public static partial class torchaudio
    {
        public static partial class datasets
        {
            public struct YesnoDatasetItem
            {
                public torch.Tensor waveform;
                public int sample_rate;
                public string[] labels;
            }
        }
    }
}