using System;
using System.Runtime.InteropServices;

namespace TorchSharp.NN
{
    public partial class Optimizer
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

            protected override bool ReleaseHandle()
            {
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
        [DllImport("LibTorchSharp")]
        extern static IntPtr NN_OptimizerAdam(IntPtr modules, int len, double learningRate);

        public static Optimizer Adam(Module module, double learningRate)
        {
            var parray = new PinnedArray<IntPtr>();
            IntPtr moduleRef;

            switch (module)
            {
                case Sequential seq:
                    var len = seq.Modules.Count;
                    moduleRef = parray.CreateArray(len);

                    for (int i = 0; i < len; i++)
                    {
                        parray.Array[i] = seq.Modules[i].handle.DangerousGetHandle();
                    }
                    
                    return new Optimizer(NN_OptimizerAdam(moduleRef, len, learningRate));
                default:
                    moduleRef = parray.CreateArray(1);
                    parray.Array[0] = module.handle.DangerousGetHandle();
                    return new Optimizer(NN_OptimizerAdam(moduleRef, 1, learningRate));
            }
            
        }

        [DllImport("LibTorchSharp")]
        extern static void NN_Optimizer_ZeroGrad(HType module);

        public void ZeroGrad()
        {
            NN_Optimizer_ZeroGrad(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static void NN_Optimizer_Step(HType module);

        public void Step()
        {
            NN_Optimizer_Step(handle);
        }
    }
}
