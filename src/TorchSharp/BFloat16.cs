// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Globalization;
using System.Runtime.InteropServices;

#nullable enable
namespace TorchSharp
{
    /// <summary>
    /// Represents a 16-bit brain floating-point number (BFloat16).
    /// Binary layout: 1 sign bit, 8 exponent bits, 7 mantissa bits — the upper 16 bits of IEEE 754 float32.
    /// Binary-compatible with c10::BFloat16 in LibTorch.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct BFloat16 : IComparable<BFloat16>, IEquatable<BFloat16>, IComparable, IFormattable
    {
        internal readonly ushort value;

        internal BFloat16(ushort rawValue, bool _)
        {
            value = rawValue;
        }

        /// <summary>
        /// Creates a BFloat16 from a float value using round-to-nearest-even (matching PyTorch c10::BFloat16).
        /// </summary>
        public BFloat16(float f)
        {
            value = FloatToBFloat16Bits(f);
        }

        /// <summary>
        /// Creates a BFloat16 from the raw 16-bit representation.
        /// </summary>
        public static BFloat16 FromRawValue(ushort rawValue) => new BFloat16(rawValue, false);

        // --- Conversion to/from float ---

        private static unsafe ushort FloatToBFloat16Bits(float f)
        {
            uint bits = *(uint*)&f;
            // NaN: preserve payload, just truncate
            if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0)
                return (ushort)(bits >> 16 | 0x0040u); // quiet NaN
            // Round-to-nearest-even (matching PyTorch c10::BFloat16)
            uint lsb = (bits >> 16) & 1u;
            uint roundingBias = 0x7FFFu + lsb;
            bits += roundingBias;
            return (ushort)(bits >> 16);
        }

        private static unsafe float BFloat16BitsToFloat(ushort raw)
        {
            int bits = raw << 16;
            return *(float*)&bits;
        }

        /// <summary>
        /// Converts this BFloat16 to a float.
        /// </summary>
        public float ToSingle() => BFloat16BitsToFloat(value);

        // --- Conversion operators ---

        public static explicit operator float(BFloat16 bf) => bf.ToSingle();
        public static explicit operator double(BFloat16 bf) => bf.ToSingle();
        public static explicit operator BFloat16(float f) => new BFloat16(f);
        public static explicit operator BFloat16(double d) => new BFloat16((float)d);

        // --- Arithmetic operators (promote to float, truncate back) ---

        public static BFloat16 operator +(BFloat16 a, BFloat16 b) => new BFloat16(a.ToSingle() + b.ToSingle());
        public static BFloat16 operator -(BFloat16 a, BFloat16 b) => new BFloat16(a.ToSingle() - b.ToSingle());
        public static BFloat16 operator *(BFloat16 a, BFloat16 b) => new BFloat16(a.ToSingle() * b.ToSingle());
        public static BFloat16 operator /(BFloat16 a, BFloat16 b) => new BFloat16(a.ToSingle() / b.ToSingle());
        public static BFloat16 operator %(BFloat16 a, BFloat16 b) => new BFloat16(a.ToSingle() % b.ToSingle());
        public static BFloat16 operator -(BFloat16 a) => new BFloat16(-a.ToSingle());

        // --- Comparison operators ---

        public static bool operator ==(BFloat16 a, BFloat16 b) => a.ToSingle() == b.ToSingle();
        public static bool operator !=(BFloat16 a, BFloat16 b) => a.ToSingle() != b.ToSingle();
        public static bool operator <(BFloat16 a, BFloat16 b) => a.ToSingle() < b.ToSingle();
        public static bool operator >(BFloat16 a, BFloat16 b) => a.ToSingle() > b.ToSingle();
        public static bool operator <=(BFloat16 a, BFloat16 b) => a.ToSingle() <= b.ToSingle();
        public static bool operator >=(BFloat16 a, BFloat16 b) => a.ToSingle() >= b.ToSingle();

        // --- IEquatable / IComparable ---

        public bool Equals(BFloat16 other) => value == other.value;
        public override bool Equals(object? obj) => obj is BFloat16 other && Equals(other);
        public override int GetHashCode() => value.GetHashCode();

        public int CompareTo(BFloat16 other) => ToSingle().CompareTo(other.ToSingle());
        public int CompareTo(object? obj)
        {
            if (obj is null) return 1;
            if (obj is BFloat16 other) return CompareTo(other);
            throw new ArgumentException("Object must be of type BFloat16.");
        }

        // --- Formatting ---

        public override string ToString() => ToSingle().ToString();
        public string ToString(string? format, IFormatProvider? formatProvider) => ToSingle().ToString(format, formatProvider);

        // --- Constants ---

        public static readonly BFloat16 Zero = FromRawValue(0x0000);
        public static readonly BFloat16 One = FromRawValue(0x3F80);
        public static readonly BFloat16 NaN = FromRawValue(0x7FC0);
        public static readonly BFloat16 PositiveInfinity = FromRawValue(0x7F80);
        public static readonly BFloat16 NegativeInfinity = FromRawValue(0xFF80);
        public static readonly BFloat16 MaxValue = FromRawValue(0x7F7F);       // ~3.39e+38
        public static readonly BFloat16 MinValue = FromRawValue(0xFF7F);       // ~-3.39e+38
        public static readonly BFloat16 Epsilon = FromRawValue(0x0080);        // smallest normal
        public static readonly BFloat16 SmallestSubnormal = FromRawValue(0x0001);

        // --- Static helpers ---

        public static bool IsNaN(BFloat16 bf) => float.IsNaN(bf.ToSingle());
        public static bool IsInfinity(BFloat16 bf) => float.IsInfinity(bf.ToSingle());
        public static bool IsPositiveInfinity(BFloat16 bf) => float.IsPositiveInfinity(bf.ToSingle());
        public static bool IsNegativeInfinity(BFloat16 bf) => float.IsNegativeInfinity(bf.ToSingle());
        public static bool IsFinite(BFloat16 bf) => !IsInfinity(bf) && !IsNaN(bf);
    }
}
