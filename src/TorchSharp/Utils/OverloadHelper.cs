using System;
using System.Collections.Generic;
using System.Linq;

namespace TorchSharp
{
    static class OverloadHelper
    {
        public static IntPtr[] ToHandleArray(this ReadOnlySpan<torch.Tensor> span)
        {
            if (span.Length == 0)
                return Array.Empty<IntPtr>();

            var result = new IntPtr[span.Length];
            for (int i = 0; i < span.Length; i++)
                result[i] = span[i].Handle;

            return result;
        }

        public static IntPtr[] ToHandleArray(this IList<torch.Tensor> list)
        {
            if (list.Count == 0)
                return Array.Empty<IntPtr>();

            var result = new IntPtr[list.Count];
            for (int i = 0; i < list.Count; i++)
                result[i] = list[i].Handle;

            return result;
        }

        public static IntPtr[] ToHandleArray(this torch.Tensor[] array) => ToHandleArray((ReadOnlySpan<torch.Tensor>)array);

        public static IntPtr[] ToHandleArray(this IEnumerable<torch.Tensor> enumerable)
        {
            return enumerable.Select(t => t.Handle).ToArray();
        }
    }
}
