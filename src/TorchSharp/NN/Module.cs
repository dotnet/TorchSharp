using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public abstract partial class Module
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

        internal HType handle;

        protected List<NN.Module> Modules { get; } = new List<NN.Module>();

        internal Module(IntPtr handle)
        {
            this.handle = new HType(handle, true);
        }

        internal Module()
        {
            this.handle = new HType(IntPtr.Zero, true);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_new_module(IntPtr names, IntPtr parameters, IntPtr with_grad, int length);

        protected Module(params Parameter[] parameters)
        {
            var names = parameters.Select(p => Marshal.StringToHGlobalAnsi(p.Name)).ToArray();
            var @params = parameters.Select(p => p.Tensor.Handle).ToArray();
            var withGrads = parameters.Select(p => p.WithGrad).ToArray();

            var namesPinned = new PinnedArray<IntPtr>();
            var paramsPinned = new PinnedArray<IntPtr>();
            var wGradPinned = new PinnedArray<bool>();

            var nparray = namesPinned.CreateArray(names);
            var pparray = paramsPinned.CreateArray(@params);
            var gparray = wGradPinned.CreateArray(withGrads);

            handle = new HType(THSNN_new_module(nparray, pparray, gparray, names.Length), true);
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
    }

    public partial class Module
    {
        static public Sequential Sequential(params Module[] modules)
        {
            return new Sequential(modules);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_linearModule(long input_size, long output_size, bool with_bias);

        static public Linear Linear(long inputSize, long outputSize, bool hasBias = false)
        {
            return new Linear(THSNN_linearModule(inputSize, outputSize, hasBias));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_conv2dModule(long inputChannel, long outputChannel, int kernelSize);

        static public Conv2D Conv2D(long inputChannel, long outputChannel, int kernelSize)
        {
            return new Conv2D(THSNN_conv2dModule(inputChannel, outputChannel, kernelSize));
        }

        static public ReLU Relu()
        {
            return new ReLU();
        }

        static public TorchTensor Relu(TorchTensor x)
        {
            return new ReLU().Forward(x);
        }

        static public MaxPool2D MaxPool2D(long kernelSize)
        {
            return new MaxPool2D(kernelSize);
        }

        static public TorchTensor MaxPool2D(TorchTensor x, long kernelSize)
        {
            using (var m = new MaxPool2D(kernelSize))
            {
                return m.Forward(x);
            }
        }

        static public LogSoftMax LogSoftMax(long dimension)
        {
            return new LogSoftMax(dimension);
        }

        static public TorchTensor LogSoftMax(TorchTensor x, long dimension)
        {
            using (var l = new LogSoftMax(dimension))
            {
                return l.Forward(x);
            }
        }

        static public Dropout Dropout(double probability, bool isTraining)
        {
            return new Dropout(probability, isTraining);
        }

        static public TorchTensor Dropout(TorchTensor x, double probability, bool isTraining)
        {
            using (var d = new Dropout(probability, isTraining))
            {
                return d.Forward(x);
            }
        }

        static public TorchTensor FeatureDropout(TorchTensor x)
        {
            using (var f = new FeatureDropout())
            {
                return f.Forward(x);
            }
        }
    }

    public abstract partial class Module : IDisposable
    {
        public abstract TorchTensor Forward(TorchTensor input);

        [DllImport("libTorchSharp")]
        extern static void THSNN_train(HType module);

        public void Train()
        {
            THSNN_train(handle);
        }

        [DllImport("libTorchSharp")]
        extern static void THSNN_eval(HType module);

        public void Eval()
        {
            THSNN_eval(handle);
        }

        [DllImport("libTorchSharp")]
        extern static bool THSNN_is_training(HType module);

        public bool IsTraining()
        {
            return THSNN_is_training(handle);
        }

        [DllImport("libTorchSharp")]
        extern static void THSNN_moduleZeroGrad(HType module);

        public virtual void ZeroGrad()
        {
            THSNN_moduleZeroGrad(handle);
        }

        [DllImport("libTorchSharp")]
        extern static void THSNN_get_named_parameters(HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

        public virtual IEnumerable<(string name, TorchTensor parameter)> NamedParameters()
        {
            // If module has no children, fetch the paramters from pytorch
            if (Modules.Any())
            {
                IEnumerable<(string name, TorchTensor parameter)> result = Enumerable.Empty<(string name, TorchTensor parameter)>();

                foreach (var module in Modules)
                {
                    result = result.Concat(module.NamedParameters());
                }

                return result;
            }

            IntPtr[] ptrArray;
            IntPtr[] strArray;

            using (var pa = new PinnedArray<IntPtr>())
            using (var sa = new PinnedArray<IntPtr>())
            {
                THSNN_get_named_parameters(handle, pa.CreateArray, sa.CreateArray);
                ptrArray = pa.Array;
                strArray = sa.Array;
            }
            return strArray.Select(s => Marshal.PtrToStringAnsi(s)).Zip(ptrArray.Select(x => new TorchTensor(x)), (x, y) => (x, y));
        }

        [DllImport("libTorchSharp")]
        extern static void THSNN_get_parameters(HType module, AllocatePinnedArray allocator);

        public virtual IEnumerable<TorchTensor > Parameters()
        {
            // If module has no children, fetch the paramters from pytorch
            if (Modules.Any())
            {
                IEnumerable<TorchTensor> result = Enumerable.Empty<TorchTensor>();

                foreach (var module in Modules)
                {
                    result = result.Concat(module.Parameters());
                }

                return result;
            }

            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                THSNN_get_parameters(handle, pa.CreateArray);
                ptrArray = pa.Array;
            }
            return ptrArray.Select(x => new TorchTensor(x));
        }

        [DllImport("libTorchSharp")]
        extern static bool THSNN_has_parameter(HType module, string name);

        public bool HasParameter(string name)
        {
            return THSNN_has_parameter(handle, name);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_get_parameter(HType module, string name);

        public TorchTensor GetParameter(string name)
        {
            var parameter = THSNN_get_parameter(handle, name);

            if (parameter == IntPtr.Zero)
            {
                throw new ArgumentNullException("Linear module without bias term.");
            }

            return new TorchTensor(parameter);
        }

        public virtual void RegisterModule(NN.Module module)
        {
            Modules.Add(module);
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
