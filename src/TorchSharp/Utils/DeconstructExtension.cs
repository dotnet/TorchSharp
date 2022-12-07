using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TorchSharp.Utils
{
    /// <summary>
    /// Converts IEnumerable to tuple.
    /// </summary>
    public static class DeconstructExtension
    {
        public static void Deconstruct<T>(this IEnumerable<T> seq, out T first, out IEnumerable<T> rest)
        {
            first = seq.FirstOrDefault();
            rest = seq.Skip(1);
        }

        public static void Deconstruct<T>(this IEnumerable<T> seq, out T first, out T second, out IEnumerable<T> rest)
            => (first, (second, rest)) = seq;

        public static void Deconstruct<T>(this IEnumerable<T> seq, out T first, out T second, out T third, out IEnumerable<T> rest)
            => (first, second, (third, rest)) = seq;

        public static void Deconstruct<T>(this IEnumerable<T> seq, out T first, out T second, out T third, out T fourth, out IEnumerable<T> rest)
            => (first, second, third, (fourth, rest)) = seq;

        public static void Deconstruct<T>(this IEnumerable<T> seq, out T first, out T second, out T third, out T fourth, out T fifth, out IEnumerable<T> rest)
            => (first, second, third, fourth, (fifth, rest)) = seq;
    }
}
