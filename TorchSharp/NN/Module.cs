using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public partial class Module
    {
        /// <summary>
        ///    Class wrapping PyTorch's module object reference.
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

        protected struct TensorPointerWrapper
        {
            public IntPtr ptr;
        }
    }

    public abstract partial class Module : IDisposable
    {
        internal HType handle;

        protected Module(IntPtr handle)
        {
            this.handle = new HType(handle, true);
        }

        ~Module()
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

        static public Sequential Sequential(params Module[] modules)
        {
            return new Sequential(modules);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr NN_linearModule(int input, int output, bool hasBias);

        static public Module Linear(int input, int output, bool hasBias = false)
        {
            return new Linear(NN_linearModule(input, output, hasBias));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr NN_reluModule();

        static public Module Relu()
        {
            return new Functional(NN_reluModule());
        }

        public abstract ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor);

        [DllImport("LibTorchSharp")]
        extern static void NN_ZeroGrad(HType module);

        public virtual void ZeroGrad()
        {
            NN_ZeroGrad(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static void NN_GetParameters(HType module, AllocatePinnedArray allocator);

        public virtual IEnumerable<ITorchTensor<float>> Parameters()
        {
            TensorPointerWrapper[] ros;

            using (var pa = new PinnedArray<TensorPointerWrapper>())
            {
                NN_GetParameters(handle, pa.CreateArray);
                ros = pa.Array;
            }
            return ros.Select(x => new FloatTensor(new FloatTensor.HType(x.ptr, true)));
        }

        [DllImport("LibTorchSharp")]
        extern static long NN_GetNumberOfChildren(HType module);

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.StdCall)]
        extern static string NN_GetModule(HType module, int index);

        public virtual string[] GetModules()
        {
            var numModules = NN_GetNumberOfChildren(handle);
            string[] result = new string[numModules];

            for (int i = 0; i < numModules; i++)
            {
                result[i] = NN_GetModule(handle, i);
            }

            return result;
        }

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.StdCall)]
        extern static string NN_GetModuleName(HType module);

        public string GetName()
        {
            return NN_GetModuleName(handle);
        }
    }
}
