// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using TorchSharp.Tensor;

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
                    public TorchTensor Tensor { get; set; }
                    public bool WithGrad { get; set; }

                    public Parameter(string name, TorchTensor parameter, bool? withGrad = null)
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
