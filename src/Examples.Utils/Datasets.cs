// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace TorchText
{
    /// <summary>
    /// This belongs in its own package, 'TorchText'.
    /// For now, it's useful to keep it with the examples that use it.
    /// </summary>
    public static class Datasets
    {
        /// <summary>
        /// WikiText2
        /// </summary>
        /// <param name="split">One of 'train', 'valid', or 'test'</param>
        /// <param name="root">The folder where the WikiText2 data set was downloaded and extracted.</param>
        /// <returns>An enumeration of lines from the text.</returns>
        /// <remarks>
        /// Download the data set at:
        /// https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
        /// </remarks>
        public static IEnumerable<string> WikiText2(string split, string root = ".data")
        {
            var dataPath = Path.Combine(root, "wikitext-2", $"wiki.{split}.tokens");
            return File.ReadLines(dataPath).Select(line => line.Trim()).Where(line => line.Length > 0);
        }

        /// <summary>
        /// WikiText2
        /// </summary>
        /// <param name="root">The folder where the WikiText2 data set was downloaded and extracted.</param>
        /// <returns>An enumeration of lines from the text for each of the data sets (training, validation, and test).</returns>
        /// <remarks>
        /// Download the data set at:
        /// https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
        /// </remarks>
        public static (IEnumerable<string>, IEnumerable<string>, IEnumerable<string>) WikiText2(string root = ".data")
        {
            return (WikiText2("train", root), WikiText2("valid", root), WikiText2("test", root));
        }
    }
}
