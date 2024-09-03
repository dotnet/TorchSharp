using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TorchSharp.Utils
{
    public class UnorderedMap<TKey, TValue> : Dictionary<TKey, TValue>, IDisposable
    {
        bool disposedValue;
        private TValue default_dict;
        //TODO: Add DefautlDict behaviour
        public UnorderedMap() { }
        private static bool IsCollectionType(Type type)
        {
            if (!type.GetGenericArguments().Any())
                return false;
            Type genericTypeDefinition = type.GetGenericTypeDefinition();
            var collectionTypes = new[] { typeof(IEnumerable<>), typeof(ICollection<>), typeof(IList<>), typeof(List<>), typeof(IList) };
            return collectionTypes.Any(x => x.IsAssignableFrom(genericTypeDefinition));
        }
        public new TValue this[TKey tk] {
            get {
                /*if (!this.ContainsKey(tk) && default_dict == null)
                    return default_dict;*/
                if (this.ContainsKey(tk))
                    return base[tk];
                var t = typeof(TValue);
                if (!IsCollectionType(t))
                    return default;
                base[tk] = (TValue)(IList)Activator.CreateInstance(typeof(List<>).MakeGenericType(t.GetGenericArguments()));
                return base[tk];
            }
            set {
                if (!this.ContainsKey(tk)) {
                    this.Add(tk, value);
                    return;
                }
                base[tk] = value;
            }
        }

        public void SetDefaultDict(TValue def)
        {
            this.default_dict = def;
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
