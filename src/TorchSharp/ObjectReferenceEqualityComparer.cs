using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace TorchSharp
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

        public new static IEqualityComparer<T> Default
        {
            get { return _defaultComparer ??= new ReferenceEqualityComparer<T>(); }
        }

        #region IEqualityComparer<T> Members

        public override bool Equals(T x, T y)
        {
            return ReferenceEquals(x, y);
        }

        public override int GetHashCode(T obj)
        {
            return RuntimeHelpers.GetHashCode(obj);
        }

        #endregion
    }
}