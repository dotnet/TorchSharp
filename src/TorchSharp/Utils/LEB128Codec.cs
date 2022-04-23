// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Collections.Generic;

namespace TorchSharp.Utils
{
    /// <summary>
    /// LEB128 encoder / decoder
    ///
    /// LEB128 is the compression format used by BinaryWriter/Reader to encode string lengths,
    /// and it is convenient to use it for other lengths in the encoding of tensors and module
    /// state dictionaries.
    /// 
    /// https://en.wikipedia.org/wiki/LEB128
    /// </summary>
    internal static class LEB128Codec
    {
        /// <summary>
        /// Encode a long value.
        /// </summary>
        /// <param name="value">The input value.</param>
        /// <returns>The encoded value as a sequence of bytes.</returns>
        public static IList<byte> Encode(long value)
        {
            if (value < 0)
                throw new NotImplementedException("LEB128 encoding of negative numbers");

            var result = new List<byte>();
            while (true) {
                long b = value & 0x7f;
                value >>= 7;
                if (value == 0) {
                    result.Add((byte)b);
                    return result;
                }
                result.Add((byte)(b | 0x80));
            }
        }

        /// <summary>
        /// Encode a long value into a binary writer.
        /// </summary>
        /// <param name="writer">A BinaryWriter instance</param>
        /// <param name="value">The input value.</param>
        public static void Encode(this BinaryWriter writer, long value)
        {
            if (value < 0)
                throw new NotImplementedException("LEB128 encoding of negative numbers");

            while (true) {
                long b = value & 0x7f;
                value >>= 7;
                if (value == 0) {
                    writer.Write((byte)b);
                    return;
                }
                writer.Write((byte)(b | 0x80));
            }
        }

        /// <summary>
        /// Decode a long value from a binary reader
        /// </summary>
        /// <param name="reader">A BinaryReader instance used for input.</param>
        /// <returns>The decoded value</returns>
        public static long Decode(this BinaryReader reader)
        {
            long result = 0;
            for (int i = 0; true; ++i) {
                long b = reader.ReadByte();
                result += ((b & 0x7f) << (i * 7));
                if ((b & 0x80) == 0) break;
            }
            return result;
        }

        /// <summary>
        /// Decode a long value from a sequence of bytes
        /// </summary>
        /// <param name="input">A sequence of bytes used for input.</param>
        /// <returns>The decoded value</returns>
        public static long Decode(IList<byte> input)
        {
            long result = 0;
            for (int i = 0; i < input.Count; ++i) {
                long b = input[i];
                result += ((b & 0x7f) << (i * 7));
                if ((b & 0x80) == 0) break;
            }
            return result;
        }
    }
}
