using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace TorchSharp.NN
{
    public class Module : IDisposable
    {
        /// <summary>
        ///    The storage class provides a mechanism to access the underlying data representation for tensors.
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

        internal Module(IntPtr handle)
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
        extern static IntPtr Module_linear(int input, int output, bool hasBias);

        static public Module Linear(int input, int output, bool hasBias = false)
        {
            return new Linear(Module_linear(input, output, hasBias));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr Module_relu();

        static public Module Relu()
        {
            return new Module(Module_relu());
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType Forward_functional(Module.HType module, FloatTensor.HType tensor);

        public virtual FloatTensor Forward(FloatTensor tensor)
        {
            return new FloatTensor(Forward_functional(handle, tensor.handle));
        }

        [DllImport("LibTorchSharp")]
        extern static void Zero_grad_functional(Module.HType module);

        public virtual void ZeroGrad()
        {
            Zero_grad_functional(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr No_grad();

        public static Module NoGrad()
        {
            return new Module(No_grad());
        }

        [DllImport("LibTorchSharp")]
        extern static void Param_functional(Module.HType module, AllocateResultOfStrategyArray allocator);

        public struct Tensor
        {
            public IntPtr ptr;
        }

        public virtual IEnumerable<FloatTensor> Parameters()
        {
            Tensor[] ros;

            using (var pa = new PinnedArray<Tensor>())
            {
                Param_functional(handle, pa.CreateArray);
                ros = pa.Array;
            }
            return ros.Select(x => new FloatTensor(new FloatTensor.HType(x.ptr, true)));
        }

        [DllImport("LibTorchSharp")]
        extern static long Get_number_of_children(Module.HType module);

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.StdCall)]
        extern static string Module_nn_get(Module.HType module, int index);

        public virtual string[] GetModules()
        {
            var numModules = Get_number_of_children(handle);
            string[] result = new string[numModules];

            for (int i = 0; i < numModules; i++)
            {
                result[i] = Module_nn_get(handle, i);
            }

            return result;
        }

        [DllImport("LibTorchSharp", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.StdCall)]
        extern static string Module_name(Module.HType module);

        public string GetName()
        {
            return Module_name(handle);
        }
    }
}
