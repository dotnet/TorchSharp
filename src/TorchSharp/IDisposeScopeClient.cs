#nullable enable
using System;

namespace TorchSharp
{
    /// <summary>
    /// Represents any object managed by the DisposeScope system. You must invoke Attach on DisposeScope manually
    /// to begin participating. Recommended implementation is to pass a DisposeScope to your object's
    /// constructor and invoke scope.Attach during creation.
    /// </summary>
    public interface IDisposeScopeClient: IDisposable
    {
        /// <summary>
        /// The DisposeScope that currently owns this object. Do not modify this property
        /// directly, it is managed by the scope system
        /// </summary>
        public DisposeScope? OwningDisposeScope { get; set; }
        /// <summary>
        /// Is true if the object has been disposed, false otherwise.
        /// </summary>
        public bool IsInvalid { get; }
    }
}