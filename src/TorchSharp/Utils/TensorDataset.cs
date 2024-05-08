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
        public class TensorDataset : torch.utils.data.IterableDataset
        {
            internal TensorDataset(torch.Tensor[] tensors)
            {
                if (tensors is null || tensors.Length == 0) throw new ArgumentNullException(nameof(tensors));
                long size1 = tensors[0].shape[0];
                if (!tensors.All(t => t.shape[0] == size1)) throw new ArgumentException("All tensors must have the same first dimension size.", nameof(tensors));

                _tensors = tensors.Select(x => x.alias().DetachFromDisposeScope()).ToArray();
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
            /// Length of the dataset, i.e. the size of the first dimension.
            /// </summary>
            public override long Count {
                get { return _tensors[0].size(0); }
            }

            public override IList<torch.Tensor> GetTensor(long index)
            {
                return this[index];
            }

            readonly torch.Tensor[] _tensors;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    foreach (var tensor in _tensors) {
                        tensor.Dispose();
                    }
                }
            }
        }
    }
}
