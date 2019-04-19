using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public partial class Optimizer : IDisposable
    {
        /// <summary>
        ///    Class wrapping PyTorch's optimzer object reference.
        /// </summary>
        internal sealed class HType : SafeHandle
        {
            public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
            {
                SetHandle(preexistingHandle);
            }

            public override bool IsInvalid => handle == IntPtr.Zero;

            // This is just for marshalling
            internal HType() : base(IntPtr.Zero, true)
            {
            }

            [DllImport("libTorchSharp")]
            extern static void THSNN_optimizerDispose(HType handle);

            protected override bool ReleaseHandle()
            {
                THSNN_optimizerDispose(this);
                return true;
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    ReleaseHandle();
                }
            }
        }

        internal HType handle;

        protected Optimizer(IntPtr handle)
        {
            this.handle = new HType(handle, true);
        }

        ~Optimizer()
        {
            Dispose(false);
        }

        /// <summary>
        ///   Releases the storage.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        protected void Dispose(bool disposing)
        {
            if (disposing)
            {
                handle.Dispose();
                handle.SetHandleAsInvalid();
            }
        }
    }

    public partial class Optimizer
    {
        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_optimizerAdam(IntPtr parameters, int len, double learningRate);

        public static Optimizer Adam(IEnumerable<ITorchTensor> parameters, double learningRate)
        {
            var parray = new PinnedArray<IntPtr>();
            IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

            return new Optimizer(THSNN_optimizerAdam(paramsRef, parray.Array.Length, learningRate));           
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_optimizerSGD(IntPtr parameters, int len, double learningRate, double momentum);

        public static Optimizer SGD(IEnumerable<ITorchTensor> parameters, double learningRate, double momentum)
        {
            var parray = new PinnedArray<IntPtr>();
            IntPtr paramsRef = parray.CreateArray(parameters.Select(p => p.Handle).ToArray());

            return new Optimizer(THSNN_optimizerSGD(paramsRef, parray.Array.Length, learningRate, momentum));
        }

        [DllImport("libTorchSharp")]
        extern static void THSNN_optimizerZeroGrad(HType module);

        public void ZeroGrad()
        {
            THSNN_optimizerZeroGrad(handle);
        }

        [DllImport("libTorchSharp")]
        extern static void THSNN_optimizerStep(HType module);

        public void Step()
        {
            THSNN_optimizerStep(handle);
        }
    }
}
