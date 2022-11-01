// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

namespace TorchSharp
{
    public static partial class torchaudio
    {
        public static partial class backend
        {
            public static partial class utils
            {
                internal static AudioBackend _backend;

                static utils()
                {
                    _backend = new NoBackend();
                }

                public static void set_audio_backend(AudioBackend backend)
                {
                    _backend = backend;
                }

                public static AudioBackend get_audio_backend()
                {
                    return _backend;
                }
            }
        }
    }
}
