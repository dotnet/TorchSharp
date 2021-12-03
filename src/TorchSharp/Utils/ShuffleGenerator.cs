// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

//Algorithm from https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/master/2017/09/18_2/VisitInDisorder.java
using System;

namespace TorchSharp.Utils
{
    class ShuffleGenerator
    {
        int maxrange;
        int prime;
        int index;
        int offset;
        int runningvalue;

        public ShuffleGenerator(int size)
        {
            var min = size / 2;
            maxrange = size;
            prime = SelectCoPrimeResev(min, size);
            offset = new Random().Next(size);
            index = 0;
            runningvalue = offset;
        }

        public bool HasNext()
        {
            return index < maxrange;
        }

        public int Next()
        {
            runningvalue += prime;
            if (runningvalue >= maxrange) runningvalue -= maxrange;
            index++;
            return runningvalue;
        }

        private const int MAX_COUNT = int.MaxValue;

        static int SelectCoPrimeResev(int min, int target)
        {
            var count = 0;
            var selected = 0;
            var rand = new Random();
            for (var val = min; val < target; ++val) {
                if (Coprime(val, target)) {
                    count += 1;
                    if ((count == 1) || (rand.Next(count) < 1)) {
                        selected = val;
                    }
                }

                if (count == MAX_COUNT) return val;
            }

            return selected;
        }

        static bool Coprime(int u, int v) => Gcd(u, v) == 1;

        static int Gcd(int u, int v)
        {
            int shift;
            if (u == 0) return v;
            if (v == 0) return u;
            for (shift = 0; ((u | v) & 1) == 0; ++shift) {
                u >>= 1;
                v >>= 1;
            }

            while ((u & 1) == 0)
                u >>= 1;

            do {
                while ((v & 1) == 0)
                    v >>= 1;
                if (u > v)
                    (v, u) = (u, v);

                v -= u;
            } while (v != 0);

            return u << shift;
        }
    }
}