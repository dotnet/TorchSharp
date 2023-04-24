// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.IO;
using System.Collections.Generic;

namespace TorchSharp
{
    using System.Data;
    using System.Linq;
    using Modules;

    public static partial class torch
    {
        public static partial class utils
        {
            public static partial class data {

                /// <summary>
                /// Dataset wrapping tensors.
                ///
                /// Each sample will be retrieved by indexing tensors along the first dimension.
                /// </summary>
                /// <param name="tensors">Tensors that have the same size of the first dimension.</param>
                public static TensorDataset TensorDataset(params torch.Tensor[] tensors) => new TensorDataset(tensors);
            }
        }
    }

    namespace Modules
    {
        public class TensorDataset : torch.utils.data.Dataset<IList<torch.Tensor>>
        {
            internal TensorDataset(torch.Tensor[] tensors)
            {
                if (tensors is null || tensors.Length == 0) throw new ArgumentNullException(nameof(tensors));
                long size1 = tensors[0].shape[0];
                if (!tensors.All(t => t.shape[0] == size1)) throw new ArgumentException("All tensors must have the same first dimension size.", nameof(tensors));

                _tensors.AddRange(tensors);
            }

            /// <summary>
            /// Indexer
            /// </summary>
            public IList<torch.Tensor> this[long index] {

                get {
                    return _tensors.Select(t => t[index]).ToList();
                }
            }

            /// <summary>
            /// Length of the dataset
            /// </summary>
            public override long Count {
                get { return _tensors.Count; }
            }

            public override IList<torch.Tensor> GetTensor(long index)
            {
                return this[index];
            }

            private List<torch.Tensor> _tensors = new List<torch.Tensor>();
        }

    }
}
