// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.torchvision.datasets;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class datasets
        {
            public static partial class utils
            {
                internal static void download_file_from_google_drive(
                    string file_id, string root, string? filename = null, string? md5 = null)
                {
                    throw new NotImplementedException();
                }

                internal static string extract_archive(string from_path, string? to_path = null, bool remove_finished = false)
                {
                    var x = torchvision.datasets.CelebA(root: "", transforms: (Tensor x, string[] y) => (2, "4"));
                    throw new NotImplementedException();
                }
            }
        }
    }
}
