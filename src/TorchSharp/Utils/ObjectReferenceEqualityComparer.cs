using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace TorchSharp.Utils
{
    /// <summary>
    /// A generic object comparerer that would only use object's reference,
    /// ignoring any <see cref="IEquatable{T}"/> or <see cref="object.Equals(object)"/>  overrides.
    /// This should be replaced with the official ReferenceEqualityComparer as soon as the TorchSharp uses .NET 6.
    /// </summary>
    public class ReferenceEqualityComparer<T> : EqualityComparer<T>
        where T : class
    {
        private static IEqualityComparer<T> _defaultComparer;
        public new static IEqualityComparer<T> Default => _defaultComparer ??= new ReferenceEqualityComparer<T>();
        public override bool Equals(T x, T y) => ReferenceEquals(x, y);
        public override int GetHashCode(T obj) => RuntimeHelpers.GetHashCode(obj);
    }
}