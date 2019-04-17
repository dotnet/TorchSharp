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
        protected sealed class HType : SafeHandle
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
            extern static void THSNN_moduleDispose(HType handle);

            protected override bool ReleaseHandle()
            {
                THSNN_moduleDispose(this);
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

        protected HType handle;

        protected bool _isTraining = true;

        protected List<NN.Module> Modules { get; } = new List<NN.Module>();

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
                foreach (var m in Modules)
                {
                    m.Dispose();
                }

                handle.Dispose();
                handle.SetHandleAsInvalid();
            }
        }
    }

    public partial class Module : IDisposable
    {
        static public Sequential Sequential(params Module[] modules)
        {
            return new Sequential(modules);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_linearModule(int input, int output, bool hasBias);

        static public Linear Linear(int input, int output, bool hasBias = false)
        {
            return new Linear(THSNN_linearModule(input, output, hasBias));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_conv2dModule(long inputChannel, long outputChannel, int kernelSize);

        static public Module Conv2D(long inputChannel, long outputChannel, int kernelSize)
        {
            return new Conv2D(THSNN_conv2dModule(inputChannel, outputChannel, kernelSize));
        }

        static public Module Relu()
        {
            return new ReLU();
        }

        static public ITorchTensor<float> Relu(ITorchTensor<float> x)
        {
            return new ReLU().Forward(x);
        }

        static public Module MaxPool2D(long kernelSize)
        {
            return new MaxPool2D(kernelSize);
        }

        static public ITorchTensor<float> MaxPool2D(ITorchTensor<float> x, long kernelSize)
        {
            using (var m = new MaxPool2D(kernelSize))
            {
                return m.Forward(x);
            }
        }

        static public Module LogSoftMax(long dimension)
        {
            return new LogSoftMax(dimension);
        }

        static public ITorchTensor<float> LogSoftMax(ITorchTensor<float> x, long dimension)
        {
            using (var l = new LogSoftMax(dimension))
            {
                return l.Forward(x);
            }
        }

        static public Module Dropout(double probability, bool isTraining)
        {
            return new Dropout(probability, isTraining);
        }

        static public ITorchTensor<float> Dropout(ITorchTensor<float> x, double probability,bool isTraining)
        {
            using (var d = new Dropout(probability, isTraining))
            {
                return d.Forward(x);
            }
        }

        static public ITorchTensor<float> FeatureDropout(ITorchTensor<float> x)
        {
            using (var f = new FeatureDropout())
            {
                return f.Forward(x);
            }
        }
    }

    public abstract partial class Module : IDisposable
    {
        public abstract ITorchTensor<float> Forward<T>(params ITorchTensor<T>[] tensors);

        public virtual void RegisterModule(NN.Module module)
        {
            Modules.Add(module);
        }

        public void Train()
        {
            _isTraining = true;
        }

        public void Eval()
        {
            _isTraining = false;
        }

        [DllImport("libTorchSharp")]
        extern static void THSNN_moduleZeroGrad(HType module);

        public virtual void ZeroGrad()
        {
            THSNN_moduleZeroGrad(handle);
        }

        public bool IsTraining()
        {
            return _isTraining;
        }

        [DllImport("libTorchSharp")]
        extern static void THSNN_getParameters(HType module, AllocatePinnedArray allocator);

        public virtual IEnumerable<ITorchTensor<float>> Parameters()
        {
            // If module has no children, fetch the paramters from pytorch
            if (Modules.Any())
            {
                IEnumerable<ITorchTensor<float>> result = Enumerable.Empty<ITorchTensor<float>>();

                foreach (var module in Modules)
                {
                    result = result.Concat(module.Parameters());
                }

                return result;
            }

            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                THSNN_getParameters(handle, pa.CreateArray);
                ptrArray = pa.Array;
            }
            return ptrArray.Select(x => new FloatTensor(x) as ITorchTensor<float>);
        }

        [DllImport("libTorchSharp")]
        extern static long THSNN_getNumberOfChildren(HType module);

        [DllImport("libTorchSharp")]
        extern static string THSNN_getChildModuleName(HType module, int index);

        public virtual IEnumerable<string> GetModules()
        {
            var numModules = THSNN_getNumberOfChildren(handle);
            string[] result = new string[numModules];

            for (int i = 0; i < numModules; i++)
            {
                result[i] = THSNN_getChildModuleName(handle, i);
            }

            return result;
        }

        [DllImport("libTorchSharp")]
        extern static string THSNN_getModuleName(HType module);

        public virtual string GetName()
        {
            return THSNN_getModuleName(handle);
        }
    }
}
