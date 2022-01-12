// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

//Algorithm from https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/master/2017/09/18_2/VisitInDisorder.java
using System;
using System.Collections;
using System.Collections.Generic;

namespace TorchSharp.Utils
{
    class FastShuffler : IEnumerable<long>
    {
        private long size;
        private int? seed;

        public FastShuffler(long size, int? seed = null)
        {
            this.size = size;
            this.seed = seed;
        }

        private class ShuffleEnumerator : IEnumerator<long>
        {
            long maxrange;
            long prime;
            long index;
            long offset;
            long runningvalue;
            private Random rand;
            private int? seed;
            private long size;

            public ShuffleEnumerator(long size, int? seed = null)
            {
                this.seed = seed;
                this.size = size;
                Reset();
            }

            private const long MAX_COUNT = long.MaxValue;

            long SelectCoPrimeResev(long min, long target)
            {
                var count = 0L;
                var selected = 0L;
                for (var val = min; val < target; ++val) {
                    if (Coprime(val, target)) {
                        count += 1;
                        if ((count == 1) || (NextLong(rand, count) < 1)) {
                            selected = val;
                        }
                    }

                    if (count == MAX_COUNT) return val;
                }

                return selected;
            }

            static bool Coprime(long u, long v) => Gcd(u, v) == 1;

            private static long Gcd(long a, long b)
            {
                while (a != 0 && b != 0)
                    if (a > b) a %= b;
                    else b %= a;

                return a | b;
            }

            private static long NextLong(Random r, long l)
            {
                var bytebuffer = new byte[8];
                r.NextBytes(bytebuffer);
                var t = 0L;
                foreach (var b in bytebuffer) {
                    t <<= 8;
                    t += b;
                }

                return t % l;
            }

            public bool MoveNext()
            {

                if (index >= maxrange) return false;
                runningvalue += prime;
                if (runningvalue >= maxrange) runningvalue -= maxrange;
                index++;
                return true;
            }

            public void Reset()
            {
                rand = seed is null ? new Random() : new Random(seed.Value);
                var min = size / 2;
                maxrange = size;
                prime = SelectCoPrimeResev(min, size);
                offset = NextLong(rand, size);
                index = 0;
                runningvalue = offset;
            }

            object IEnumerator.Current => Current;

            public long Current => runningvalue;

            public void Dispose()
            {

            }
        }

        public IEnumerator<long> GetEnumerator()
        {
            return new ShuffleEnumerator(size, seed);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}