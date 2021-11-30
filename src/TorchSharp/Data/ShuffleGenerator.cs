using System;

namespace TorchSharp.Data
{
    public class ShuffleGenerator
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
            prime = selectCoPrimeResev(min, size);
            offset = new Random().Next(size);
            index = 0;
            runningvalue = offset;
        }

        private int getCurrentValue()
        {
            return (int) (((long) index * prime + offset) % (maxrange));
        }

        public bool hasNext()
        {
            return index < maxrange;
        }

        public int next()
        {
            runningvalue += prime;
            if (runningvalue >= maxrange) runningvalue -= maxrange;
            index++;
            return runningvalue;
        }

        private const int MAX_COUNT = int.MaxValue;

        static int selectCoPrimeResev(int min, int target)
        {
            var count = 0;
            var selected = 0;
            var rand = new Random();
            for (var val = min; val < target; ++val) {
                if (coprime(val, target)) {
                    count += 1;
                    if ((count == 1) || (rand.Next(count) < 1)) {
                        selected = val;
                    }
                }

                if (count == MAX_COUNT) return val;
            }

            return selected;
        }

        static bool coprime(int u, int v) => gcd(u, v) == 1;

        static int gcd(int u, int v)
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