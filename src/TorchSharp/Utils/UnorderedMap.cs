using System;
using System.Collections.Generic;
using System.Text;

namespace TorchSharp.Utils
{
    public class UnorderedMap<TKey, TValue> : Dictionary<TKey, TValue>, IDisposable
    {
        bool disposedValue;

        public UnorderedMap() { }
        public new TValue this[TKey tk] {
            get {
                if (this.ContainsKey(tk))
                    return base[tk];
                return default(TValue);
            }
            set {
                if (!this.ContainsKey(tk)) {
                    this.Add(tk, value);
                    return;
                }
                base[tk] = value;
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue) {
                if (disposing) {
                    base.Clear();
                    // TODO: dispose managed state (managed objects)
                }

                // TODO: free unmanaged resources (unmanaged objects) and override finalizer
                // TODO: set large fields to null
                disposedValue = true;
            }
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~UnorderedMap()
        // {
        //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        //     Dispose(disposing: false);
        // }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
