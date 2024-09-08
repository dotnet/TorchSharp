using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TorchSharp.Utils
{
    public class Dictionary<TKey1, TKey2, TValue> : Dictionary<Tuple<TKey1, TKey2>, TValue>, IDictionary<Tuple<TKey1, TKey2>, TValue>
    {

        public TValue this[TKey1 key1, TKey2 key2] {
            get { return base[Tuple.Create(key1, key2)]; }
            set { base[Tuple.Create(key1, key2)] = value; }
        }

        public void Add(TKey1 key1, TKey2 key2, TValue value)
        {
            base.Add(Tuple.Create(key1, key2), value);
        }

        public bool ContainsKey(TKey1 key1, TKey2 key2)
        {
            return base.ContainsKey(Tuple.Create(key1, key2));
        }
    }

    public class UnorderedMap<TKey1, TKey2, TValue> : Dictionary<TKey1, TKey2, TValue>, IDisposable
    {
        bool disposedValue;
        public new TValue this[TKey1 tk1, TKey2 tk2] {
            get {
                /*if (!this.ContainsKey(tk) && default_dict == null)
                    return default_dict;*/
                if (this.ContainsKey(tk1, tk2))
                    return base[tk1, tk2];
                return default;
            }
            set {
                if (!this.ContainsKey(tk1, tk2)) {
                    this.Add(tk1, tk2, value);
                    return;
                }
                base[tk1, tk2] = value;
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
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
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
