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
            public static partial class utils
            {
                internal static IAudioBackend _backend;

                static utils()
                {
                    _backend = new NoBackend();
                }

                public static void set_audio_backend(IAudioBackend backend)
                {
                    _backend = backend;
                }

                public static IAudioBackend get_audio_backend()
                {
                    return _backend;
                }
            }
        }
    }
}
