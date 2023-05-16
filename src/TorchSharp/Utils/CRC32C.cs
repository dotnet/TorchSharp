// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// Note: the native implementation of CRC32C has the following copyright / license notice:

/* Part of CRC-32C library: https://crc32c.machinezoo.com/ */
/*
  Copyright (c) 2013 - 2014, 2016 Mark Adler, Robert Vazan, Max Vysokikh

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the author be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
  claim that you wrote the original software. If you use this software
  in a product, an acknowledgment in the product documentation would be
  appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
  misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

using System;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp.Utils
{
    public static class CRC32C
    {
        /// <summary>
        /// Compule the CRC32C value for a byte array.
        /// </summary>
        /// <param name="data">A byte array.</param>
        public static unsafe uint process(byte[] data)
        {
            fixed (byte* ptr = data) {
                return crc32c_append(0, (IntPtr)ptr, (ulong)data.Length);
            }
        }

        /// <summary>
        /// Compule the CRC32C value for a 32-bit integer.
        /// </summary>
        /// <param name="data">A byte array.</param>
        public static unsafe uint process(int data)
        {
            fixed (int* ptr = stackalloc int[] { data })
                return crc32c_append(0, (IntPtr)ptr, (ulong)sizeof(int));
        }

        /// <summary>
        /// Compule the CRC32C value for a 64-bit integer.
        /// </summary>
        /// <param name="data">A byte array.</param>
        public static unsafe uint process(long data)
        {
            fixed (long* ptr = stackalloc long[] { data })
                return crc32c_append(0, (IntPtr)ptr, (ulong)sizeof(long));
        }
    }
}
