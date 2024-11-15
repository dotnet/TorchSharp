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
        public class ConcatDataset<T> : IDataset<T>
        {
            private static IEnumerable<long> Cumsum(IEnumerable<IDataset<T>> datasets)
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

            // Here we have to use arrays since the given index is in Int64...
            private readonly IDataset<T>[] _datasets;
            public IReadOnlyList<IDataset<T>> datasets => _datasets;

            private readonly long[] _cumulativeSizes;
            public IReadOnlyList<long> cumulative_sizes => _cumulativeSizes;

            private readonly bool leaveOpen;

            public ConcatDataset(
                IEnumerable<IDataset<T>> datasets,
                bool leaveOpen = false)
            {
                this._datasets = datasets.ToArray();
                if (this._datasets.Length is 0)
                    throw new ArgumentException(
                        "datasets should not be an empty iterable", nameof(datasets));

                // PyTorch also says 'ConcatDataset does not support IterableDataset'.
                // But it's not our torch.utils.data.IterableDataset in TorchSharp.
                this._cumulativeSizes = Cumsum(datasets).ToArray();

                this.leaveOpen = leaveOpen;
            }

            public long Count => this._cumulativeSizes.Last();

            public T this[long index]
            {
                get {
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
            }

            public void Dispose()
            {
                if (!leaveOpen) {
                    foreach (var dataset in this._datasets)
                        dataset.Dispose();
                }
            }
        }
    }
}