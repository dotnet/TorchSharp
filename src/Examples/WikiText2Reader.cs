using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using TorchSharp.Tensor;

namespace TorchText
{
    public static class Datasets
    {
        public static IEnumerable<string> WikiText2(string split, string root = ".data")
        {
            var dataPath = Path.Combine(root, "wikitext-2", $"wiki.{split}.tokens");
            return File.ReadLines(dataPath).Select(line => line.Trim()).Where(line => line.Length > 0);
        }
        public static (IEnumerable<string>, IEnumerable<string>, IEnumerable<string>) WikiText2(string root = ".data")
        {
            return (WikiText2("train", root), WikiText2("valid", root), WikiText2("test", root));
        }
    }
}
