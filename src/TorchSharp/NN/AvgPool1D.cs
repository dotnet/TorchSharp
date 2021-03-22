// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a AvgPool1D module.
    /// </summary>
    public class AvgPool1D : Module
    {
        internal AvgPool1D (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_AvgPool1d_forward (IntPtr module, IntPtr tensor);

        public TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_AvgPool1d_forward (handle.DangerousGetHandle (), tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_AvgPool1d_ctor (IntPtr pkernelSize, IntPtr pstrides,  out IntPtr pBoxedModule);

        static public AvgPool1D AvgPool1D(long kernelSize, long? stride = null)
        {
            return stride.HasValue ?
                AvgPool1D(new long[] { kernelSize }, new long[] { stride.Value }) :
                AvgPool1D(new long[] { kernelSize }, null);
        }

        static private AvgPool1D AvgPool1D (long[] kernelSize, long[] strides = null)
        {
            unsafe {
                fixed (long* pkernelSize = kernelSize, pstrides = strides) {
                    var handle = THSNN_AvgPool1d_ctor ((IntPtr)pkernelSize, (IntPtr)pstrides, out var boxedHandle);
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new AvgPool1D (handle, boxedHandle);
                }
            }
        }
    }

    public static partial class Functions
    {
        static public TorchTensor AvgPool1D (TorchTensor x, long kernelSize, long? stride = null)
        {
            using (var d = Modules.AvgPool1D (kernelSize, stride)) {
                return d.forward (x);
            }
        }
    }
}
