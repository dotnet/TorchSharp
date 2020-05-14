// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;

namespace TorchSharp
{
    internal static class ExtensionMethods
    {
        internal static void Deconstruct<T>(this IList<T> list, out T head, out IList<T> tail)
        {
            head = list.FirstOrDefault();
            tail = new List<T>(list.Skip(1));
        }
    }
}
