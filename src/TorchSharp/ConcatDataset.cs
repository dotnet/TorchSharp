// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp.Modules;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class utils
        {
            public static partial class data
            {
                public static ConcatDataset<T> ConcatDataset<T>(IEnumerable<IDataset<T>> datasets)
                {
                    return new ConcatDataset<T>(datasets);
                }
            }
        }
    }

    namespace Modules
    {
        public class ConcatDataset<T> : torch.utils.data.Dataset<T>
        {
            private static IEnumerable<long> Cumsum(
                IEnumerable<torch.utils.data.IDataset<T>> datasets)
            {
                var s = 0L;
                foreach (var e in datasets) {
                    s += e.Count;
                    yield return s;
                }
            }
            private static long bisectRight(long[] a, long x)
            {
                var lo = 0;
                var hi = a.Length;
                while (lo < hi) {
                    var mid = (lo + hi) / 2;
                    if (x < a[mid])
                        hi = mid;
                    else
                        lo = mid + 1;
                }
                return lo;
            }


            private readonly torch.utils.data.IDataset<T>[] _datasets;
            public IReadOnlyList<torch.utils.data.IDataset<T>> datasets => _datasets;

            private readonly long[] _cumulativeSizes;
            public IReadOnlyList<long> cumulative_sizes => _cumulativeSizes;

            private readonly bool autoDispose;

            public ConcatDataset(
                IEnumerable<torch.utils.data.IDataset<T>> datasets,
                bool autoDispose = true)
            {
                this._datasets = datasets.ToArray();
                if (this._datasets.Length is 0)
                    throw new ArgumentException(
                        "datasets should not be an empty iterable", nameof(datasets));

                // PyTorch also says 'ConcatDataset does not support IterableDataset'.
                // But it's not our torch.utils.data.IterableDataset in TorchSharp.
                this._cumulativeSizes = Cumsum(datasets).ToArray();

                this.autoDispose = autoDispose;
            }

            public override long Count => this._cumulativeSizes.Last();

            public override T GetTensor(long index)
            {
                if (index < 0) {
                    if (-index > this.Count) {
                        throw new ArgumentException(
                            "absolute value of index should not exceed dataset length",
                            nameof(index));
                    }
                    index = this.Count + index;
                }

                var datasetIdx = bisectRight(this._cumulativeSizes, index);
                long sampleIdx;
                if (datasetIdx == 0)
                    sampleIdx = index;
                else
                    sampleIdx = index - this._cumulativeSizes[datasetIdx - 1];
                return this._datasets[datasetIdx][sampleIdx];
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing && autoDispose) {
                    foreach (var dataset in this._datasets)
                        dataset.Dispose();
                }

                base.Dispose(disposing);
            }
        }
    }
}