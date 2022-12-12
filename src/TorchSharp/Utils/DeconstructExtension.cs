using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TorchSharp.Utils
{
    /// <summary>
    /// Converts IEnumerable to tuple.
    /// Example:
    /// int[] rect = new int[4];
    /// var (left, top, width, height, _) = rect;
    /// </summary>
    public static class DeconstructExtension
    {
        /// <summary>
        /// Deconstructs a sequence to the first element and rest of elements.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="seq"></param>
        /// <param name="first"></param>
        /// <param name="rest"></param>
        public static void Deconstruct<T>(this IEnumerable<T> seq, out T first, out IEnumerable<T> rest)
        {
            first = seq.FirstOrDefault();
            rest = seq.Skip(1);
        }

        /// <summary>
        /// Deconstrcts one element out of sequence.
        /// </summary>
        public static void Deconstruct<T>(this IEnumerable<T> seq, out T first, out T second, out IEnumerable<T> rest)
            => (first, (second, rest)) = seq;

        /// <summary>
        /// Deconstrcts two elements out of sequence.
        /// </summary>
        public static void Deconstruct<T>(this IEnumerable<T> seq, out T first, out T second, out T third, out IEnumerable<T> rest)
            => (first, second, (third, rest)) = seq;

        /// <summary>
        /// Deconstrcts three elements out of sequence.
        /// </summary>
        public static void Deconstruct<T>(this IEnumerable<T> seq, out T first, out T second, out T third, out T fourth, out IEnumerable<T> rest)
            => (first, second, third, (fourth, rest)) = seq;

        /// <summary>
        /// Deconstrcts four elements out of sequence.
        /// </summary>
        public static void Deconstruct<T>(this IEnumerable<T> seq, out T first, out T second, out T third, out T fourth, out T fifth, out IEnumerable<T> rest)
            => (first, second, third, fourth, (fifth, rest)) = seq;
    }
}
