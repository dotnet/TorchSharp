// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Collections;

#nullable enable
namespace TorchSharp
{
    public static partial class torch
    {
        public struct Size : IEnumerable<long>
        {
            public Size(long[] shape)
            {
                _shape = shape;
            }

            public Size(long size)
            {
                _shape = new long[] { size };
            }

            public Size((long,long) size)
            {
                _shape = new long[] { size.Item1, size.Item2 };
            }

            public Size((long, long, long) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3 };
            }
            public Size((long, long, long, long) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3, size.Item4 };
            }
            public Size((long, long, long, long, long) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3, size.Item4, size.Item5 };
            }
            public Size((long, long, long, long, long, long) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3, size.Item4, size.Item5, size.Item6 };
            }

            public Size(int size)
            {
                _shape = new long[] { size };
            }

            public Size((int, int) size)
            {
                _shape = new long[] { size.Item1, size.Item2 };
            }

            public Size((int, int, int) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3 };
            }
            public Size((int, int, int, int) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3, size.Item4 };
            }
            public Size((int, int, int, int, int) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3, size.Item4, size.Item5 };
            }
            public Size((int, int, int, int, int, int) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3, size.Item4, size.Item5, size.Item6 };
            }

            public static implicit operator Size(long size) => new Size(size);
            public static implicit operator Size(long[] size) => new Size(size);
            public static implicit operator Size((long, long) size) => new Size(size);
            public static implicit operator Size((long, long, long) size) => new Size(size);
            public static implicit operator Size((long, long, long, long) size) => new Size(size);
            public static implicit operator Size((long, long, long, long, long) size) => new Size(size);
            public static implicit operator Size((long, long, long, long, long, long) size) => new Size(size);

            //public static implicit operator long[](Size size) => size._shape;

            public static bool operator ==(Size left, Size right) => left._shape == right._shape;

            public static bool operator !=(Size left, Size right) => left._shape != right._shape;

            public override bool Equals(object? obj)
            {
                if (obj is null || !(obj is Size)) return false;

                return _shape.Equals(((Size)obj)._shape);
            }

            public override int GetHashCode()
            {
                return _shape.GetHashCode();
            }

            public int Length => _shape.Length;

            public IEnumerable<long> Take(int i) => _shape.Take<long>(i);

            public IEnumerator<long> GetEnumerator()
            {
                return _shape.AsEnumerable<long>().GetEnumerator();
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return this.GetEnumerator();
            }

            public long this[int idx] {
                get { return _shape[idx]; }
            }

            public long this[long idx] {
                get { return _shape[(int)idx]; }
            }

            private long[] _shape;
        }
    }
}
