// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
namespace TorchSharp
{
    public enum compute_mode
    {
        use_mm_for_euclid_dist_if_necessary = 0,
        use_mm_for_euclid_dist = 1,
        donot_use_mm_for_euclid_dist = 2
    }
}