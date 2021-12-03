// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

//Algorithm from https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/master/2017/09/18_2/VisitInDisorder.java
using System;

namespace TorchSharp.Utils
{
    class ShuffleGenerator
    {
        long maxrange;
        long prime;
        long index;
        long offset;
        long runningvalue;
        private Random rand;
        public ShuffleGenerator(long size, int? seed = null)
        {
            rand = seed is null ? new Random() : new Random(seed.Value);
            var min = size / 2;
            maxrange = size;
            prime = SelectCoPrimeResev(min, size);
            offset = NextLong(rand, size);
            index = 0;
            runningvalue = offset;
        }

        public bool HasNext()
        {
            return index < maxrange;
        }

        public long Next()
        {
            runningvalue += prime;
            if (runningvalue >= maxrange) runningvalue -= maxrange;
            index++;
            return runningvalue;
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
    }
}