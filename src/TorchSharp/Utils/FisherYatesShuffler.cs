using System;
using System.Collections;
using System.Collections.Generic;

namespace TorchSharp.Utils
{
    public class FisherYatesShuffler : IEnumerable<long>
    {
        private long size;
        private int? seed;

        public FisherYatesShuffler(long size, int? seed = null)
        {
            this.size = size;
            this.seed = seed;
        }

        public IEnumerator<long> GetEnumerator()
            => seed == null
                ? new FisherYatesShufflerEnumerable(size)
                : new FisherYatesShufflerEnumerable(size, seed.Value);

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        private class FisherYatesShufflerEnumerable : IEnumerator<long>
        {
            public FisherYatesShufflerEnumerable(long size)
            {
                this.size = size;
                this.seed = null;
                Reset();
            }

            public FisherYatesShufflerEnumerable(long size, int seed)
            {
                this.size = size;
                this.seed = seed;
                Reset();
            }

            private int? seed;
            private long size;
            private Dictionary<long, long> indexs = new();
            private Random r;
            private long current = -1;

            public bool MoveNext()
            {
                current++;
                return current < size;
            }

            public void Reset()
            {
                r = seed == null ? new Random() : new Random(seed.Value);
                current = -1;
                indexs.Clear();
                for (var i = 0L; i < size; i++) {
                    var rndidx = GetRandomLong(i);
                    if (rndidx == i)
                        indexs[i] = i;
                    else {
                        indexs[i] = indexs[rndidx];
                        indexs[rndidx] = i;
                    }
                }
            }

            public long Current => indexs[current];

            object IEnumerator.Current => Current;

            public void Dispose()
            {
                /* Ignore */
            }

            private long GetRandomLong(long l)
            {
                unchecked {
                    return (((long) r.Next() << 32) + (uint) r.Next()) % (l + 1);
                }
            }
        }
    }
}