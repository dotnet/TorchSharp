// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using TorchSharp;

using static TorchSharp.torch.nn;

namespace TorchText.Vocab
{
    /// <summary>
    /// This needs a permanent place.
    /// The functionality is based on the Python 'Counter' class.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Counter<T> : IEnumerable<KeyValuePair<T, int>>
    {
        private Dictionary<T, int> _dict = new Dictionary<T, int>();

        public void update(T key)
        {
            if (_dict.TryGetValue(key, out int count)) {
                _dict[key] = count + 1;
            } else {
                _dict[key] = 1;
            }
        }
        public void update(IEnumerable<T> keys)
        {
            foreach (T key in keys) {
                update(key);
            }
        }
        public int this[T key] { get => _dict[key]; }

        public IEnumerator<KeyValuePair<T, int>> GetEnumerator()
        {
            return _dict.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    /// <summary>
    /// This belongs in its own package, 'TorchText'.
    /// For now, it's useful to keep it with the examples that use it.
    /// </summary>
    public class Vocab
    {
        public Vocab(Counter<string> counter, int? maxSize = null, int minFreq = 1, string[] specials = null, Func<torch.Tensor, torch.Tensor> unkInit = null, bool specialsFirst = true)
        {
            if (specials == null) specials = new string[] { "<unk>", "<pad>" };
            if (unkInit == null) unkInit = (t => init.zeros_(t.clone()));
            if (specialsFirst) {
                foreach (var sp in specials) {
                    _dict.Add(sp, _last++);
                }
            }
            foreach (var kv in counter.Where(kv => kv.Value >= minFreq)) {
                if (!specials.Contains(kv.Key)) {
                    _dict.Add(kv.Key, _last++);
                }
                if (_last > (maxSize ?? int.MaxValue))
                    break;
            }
            if (!specialsFirst) {
                foreach (var sp in specials) {
                    _dict.Add(sp, _last++);
                }
            }
        }

        public int this[string key] { get => _dict.TryGetValue(key, out int value) ? value : _dict["<unk>"]; }

        public int Count => _dict.Count;

        public void Add(string key, int value)
        {
            _dict.Add(key, value);
        }

        public void Add(KeyValuePair<string, int> item)
        {
            Add(item.Key, item.Value);
        }

        public bool TryGetValue(string key, [MaybeNullWhen(false)] out int value)
        {
            return _dict.TryGetValue(key, out value);
        }

        private Dictionary<string, int> _dict = new Dictionary<string, int>();
        private int _last = 0;
    }
}
