//NOTE: This make compatibility of Range with NetStandard2.0 may need include System.Runtime.InteropServices.RuntimeInformation
/*
#if NETSTANDARD2_0
#region License
// MIT License
// 
// Copyright (c) Manuel Römer
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#endregion

namespace System.Runtime.CompilerServices
{
    public static class RuntimeHelpers
    {
        public static T[] GetSubArray<T>(T[] array, Range range)
        {
            var (offset, length) = range.GetOffsetAndLength(array.Length);
            if (length == 0)
                return Array.Empty<T>();
            T[] dest;
            if (typeof(T).IsValueType || typeof(T[]) == array.GetType()) {
                // We know the type of the array to be exactly T[] or an array variance
                // compatible value type substitution like int[] <-> uint[].

                if (length == 0) {
                    return Array.Empty<T>();
                }

                dest = new T[length];
            } else {
                // The array is actually a U[] where U:T. We'll make sure to create
                // an array of the exact same backing type. The cast to T[] will
                // never fail.

                dest = (T[])(Array.CreateInstance(array.GetType().GetElementType()!, length));
            }
            Array.Copy(array, offset, dest, 0, length);
            return dest;
        }
    }
}
#endif*/