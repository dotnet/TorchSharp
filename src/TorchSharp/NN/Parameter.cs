// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using static TorchSharp.torch;

namespace TorchSharp
{

    public static partial class torch
    {
        public static partial class nn
        {
            public static partial class parameter
            {
                public struct Parameter
                {
                    public string Name { get; set; }
                    public Tensor Tensor { get; set; }
                    public bool WithGrad { get; set; }

                    public Parameter(string name, Tensor parameter, bool? withGrad = null)
                    {
                        Name = name;
                        Tensor = parameter;
                        WithGrad = withGrad ?? parameter.requires_grad;
                    }
                };
            }
        }
    }
}
