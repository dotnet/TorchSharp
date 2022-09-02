// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

namespace TorchSharp
{
    public static partial class torchaudio
    {
        public static partial class models
        {
            /// <summary>
            /// Normalization mode of feature extractor
            /// </summary>
            public enum FeatureExtractorNormMode
            {
                group_norm,
                layer_norm
            }
        }
    }
}
