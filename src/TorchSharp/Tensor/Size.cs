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
        /// <summary>
        /// Represents the dimensions of a tensor.
        /// </summary>
        /// <remarks>
        /// The primary purpose of this type, at the moment, is to avoid having to declare
        /// too many overloads on tensor factories that take input sizes.
        /// The name was chosen to coincide with 'torch.Size' in PyTorch. It may later be
        /// used as the return value of 'Tensor.shape' and 'Tensor.size()'
        /// </remarks>
        public struct Size : IEnumerable<long>
        {
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="shape">An array of longs, the size of an N-D tensor.</param>
            public Size(long[] shape)
            {
                _shape = shape;
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="shape">An array of longs, the size of an N-D tensor.</param>
            public Size(int[] shape)
            {
                _shape = shape.Select(i => (long)i).ToArray();
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="size">The size of a 1D tensor.</param>
            public Size(long size)
            {
                _shape = new long[] { size };
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="size">The size of a 2D tensor.</param>
            public Size((long, long) size)
            {
                _shape = new long[] { size.Item1, size.Item2 };
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="size">The size of a 3D tensor.</param>
            public Size((long, long, long) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3 };
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="size">The size of a 4D tensor.</param>
            public Size((long, long, long, long) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3, size.Item4 };
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="size">The size of a 5D tensor.</param>
            public Size((long, long, long, long, long) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3, size.Item4, size.Item5 };
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="size">The size of a 6D tensor.</param>
            public Size((long, long, long, long, long, long) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3, size.Item4, size.Item5, size.Item6 };
            }


            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="size">The size of a 1D tensor.</param>
            public Size(int size)
            {
                _shape = new long[] { size };
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="size">The size of a 2D tensor.</param>
            public Size((int, int) size)
            {
                _shape = new long[] { size.Item1, size.Item2 };
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="size">The size of a 3D tensor.</param>
            public Size((int, int, int) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3 };
            }
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="size">The size of a 4D tensor.</param>
            public Size((int, int, int, int) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3, size.Item4 };
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="size">The size of a 5D tensor.</param>
            public Size((int, int, int, int, int) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3, size.Item4, size.Item5 };
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="size">The size of a 6D tensor.</param>
            public Size((int, int, int, int, int, int) size)
            {
                _shape = new long[] { size.Item1, size.Item2, size.Item3, size.Item4, size.Item5, size.Item6 };
            }

            /// <summary>
            /// Implicit conversion operators. Useful to avoid overloads everywhere.
            /// </summary>
            public static implicit operator Size(long size) => new Size(size);
            public static implicit operator Size((long, long) size) => new Size(size);
            public static implicit operator Size((long, long, long) size) => new Size(size);
            public static implicit operator Size((long, long, long, long) size) => new Size(size);
            public static implicit operator Size((long, long, long, long, long) size) => new Size(size);
            public static implicit operator Size((long, long, long, long, long, long) size) => new Size(size);

            public static Size operator +(Size left, Size right)
            {
                return new Size(left._shape.AsEnumerable<long>().Concat<long>(right._shape).ToArray());
            }

            //public static implicit operator Size(long[] size) => new Size(size);
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

            public long numel()
            {
                long size = 1;
                foreach (var s in _shape) {
                    size *= s;
                }
                return size;
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

            internal long[] Shape { get { return _shape; } }

            private long[] _shape;
        }
    }
}
