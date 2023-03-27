using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;

namespace TorchSharp.Utils
{
    public class OrderedDict<TKey, TValue> : IDictionary<TKey, TValue>, IList<(TKey, TValue)>
    {
        /// <summary>
        /// Remove all items from the ParameterDict.
        /// </summary>
        public void clear()
        {
            _list.Clear();
            _dict.Clear();
        }

        /// <summary>
        /// Return an enumeration of the ParameterDict key/value pairs.
        /// </summary>
        /// <returns></returns>
        public IEnumerator<(TKey, TValue)> items() => _list.GetEnumerator();

        /// <summary>
        /// Return the ParameterDict keys.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<TKey> keys() => _dict.Keys;

        /// <summary>
        /// Return the ParameterDict values.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<TValue> values() => _dict.Values;

        public (TKey, TValue) this[int index] {
            get => _list[index];
            set {
                var key = value.Item1;
                _list[index] = value;
                _dict[key] = value.Item2;
            }
        }

        public bool IsReadOnly => false;

        public ICollection<TKey> Keys => _list.Select(kv => kv.Item1).ToList();

        public ICollection<TValue> Values => _list.Select(kv => kv.Item2).ToList();

        public int Count => _dict.Count;

        public TValue this[TKey key] {
            get => _dict[key];
            set {
                _dict[key] = value;
                var idx = _list.FindIndex(kv => kv.Item1.Equals(key));
                if (idx >= 0) {
                    _list[idx] = (key, value);
                }
                else {
                    _list.Add((key, value));
                }
            }
        }

        public void Add((TKey, TValue) item)
        {
            _dict.Add(item.Item1, item.Item2);
            _list.Add(item);
        }

        public void Add(TKey key, TValue value)
        {
            _dict.Add(key, value);
            _list.Add((key, value));
        }

        public void Add(KeyValuePair<TKey, TValue> item)
        {
            _dict.Add(item.Key, item.Value);
            _list.Add((item.Key, item.Value));
        }

        public bool Contains((TKey, TValue) item)
        {
            return _list.Contains(item);
        }

        public void CopyTo((TKey, TValue)[] array, int arrayIndex)
        {
            _list.CopyTo(array, arrayIndex);
        }

        public int IndexOf((TKey, TValue) item)
        {
            return _list.IndexOf(item);
        }

        public void Insert(int index, (TKey, TValue) item)
        {
            _dict.Add(item.Item1, item.Item2);
            _list.Insert(index, item);
        }

        public bool Remove((TKey, TValue) item)
        {
            _dict.Remove(item.Item1);
            return _list.Remove(item);
        }

        public void RemoveAt(int index)
        {
            if (index >= _list.Count) throw new IndexOutOfRangeException();
            var (n, p) = _list[index];
            _list.RemoveAt(index);
            _dict.Remove(n);
        }

        public bool ContainsKey(TKey key)
        {
            return _dict.ContainsKey(key);
        }

        public bool Remove(TKey key)
        {
            var value = _dict[key];
            var idx = _list.FindIndex(kv => kv.Item1.Equals(key));
            _list.RemoveAt(idx);
            return _dict.Remove(key);
        }

        public bool TryGetValue(TKey key, [MaybeNullWhen(false)] out TValue value)
        {
            return _dict.TryGetValue(key, out value);
        }

        public void Clear()
        {
            _dict.Clear();
            _list.Clear();
        }

        public bool Contains(KeyValuePair<TKey, TValue> item)
        {
            return _dict.ContainsKey(item.Key);
        }

        public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
        {
            throw new NotImplementedException();
        }

        public bool Remove(KeyValuePair<TKey, TValue> item)
        {
            return _dict.Remove(item.Key);
        }

        public IEnumerator<(TKey, TValue)> GetEnumerator()
        {
            return _list.GetEnumerator();
        }

        IEnumerator<KeyValuePair<TKey, TValue>> IEnumerable<KeyValuePair<TKey, TValue>>.GetEnumerator()
        {
            throw new NotImplementedException();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable<KeyValuePair<TKey, TValue>>)this).GetEnumerator();
        }

        private List<(TKey, TValue)> _list = new List<(TKey, TValue)>();
        private Dictionary<TKey, TValue> _dict = new Dictionary<TKey, TValue>();
    }

}
