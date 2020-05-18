//// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
//using System;
//using System.Linq;
//using System.Runtime.InteropServices;
//using TorchSharp.Tensor;

//namespace TorchSharp.JIT
//{
//    public class Module : IDisposable
//    {
//        /// <summary>
//        ///    Class wrapping PyTorch's module object reference.
//        /// </summary>
//        internal sealed class HType : SafeHandle
//        {
//            public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
//            {
//                SetHandle(preexistingHandle);
//            }

//            public override bool IsInvalid => handle == IntPtr.Zero;

//            // This is just for marshalling
//            internal HType() : base(IntPtr.Zero, true)
//            {
//            }

//            [DllImport("LibTorchSharp")]
//            private static extern void THSJIT_moduleDispose(HType handle);

//            protected override bool ReleaseHandle()
//            {
//                THSJIT_moduleDispose(this);
//                return true;
//            }

//            protected override void Dispose(bool disposing)
//            {
//                if (disposing)
//                {
//                    ReleaseHandle();
//                }
//            }
//        }

//        internal HType handle;

//        internal Module(IntPtr handle)
//        {
//            this.handle = new HType(handle, true);
//        }

//        ~Module()
//        {
//            Dispose(false);
//        }

//        /// <summary>
//        ///   Releases the storage.
//        /// </summary>
//        public void Dispose()
//        {
//            Dispose(true);
//            GC.SuppressFinalize(this);
//        }

//        /// <summary>
//        ///   Implements the .NET Dispose pattern.
//        /// </summary>
//        protected void Dispose(bool disposing)
//        {
//            if (disposing)
//            {
//                handle.Dispose();
//                handle.SetHandleAsInvalid();
//            }
//        }

//        [DllImport("LibTorchSharp")]
//        private static extern IntPtr THSJIT_loadModule(string filename);

//        static public Module Load(string filename)
//        {
//            return new Module(THSJIT_loadModule(filename));
//        }

//        [DllImport("LibTorchSharp")]
//        private static extern long THSJIT_getNumModules(HType module);

//        [DllImport("LibTorchSharp")]
//        private static extern int THSJIT_getNumberOfInputs(HType module);

//        public int GetNumberOfInputs()
//        { 
//            return THSJIT_getNumberOfInputs(handle);
//        }

//        [DllImport("LibTorchSharp")]
//        private static extern int THSJIT_getNumberOfOutputs(HType module);

//        public int GetNumberOfOutputs()
//        {
//            return THSJIT_getNumberOfOutputs(handle);
//        }

//        [DllImport("LibTorchSharp")]
//        private static extern IntPtr THSJIT_getInputType(HType module, int index);

//        public Type GetInputType(int index)
//        {
//            var type = new Type(THSJIT_getInputType(handle, index));

//            return GetType(type);
//        }

//        [DllImport("LibTorchSharp")]
//        private static extern IntPtr THSJIT_getOutputType(HType module, int index);

//        public Type GetOutputType(int index)
//        {
//            var type = new Type(THSJIT_getOutputType(handle, index));

//            return GetType(type);
//        }

//        private Type GetType(Type type)
//        {
//            switch (type.Kind)
//            {
//                case Type.TypeKind.TensorType:
//                    var dynamic = type.AsDynamicType();
//                    type.Dispose();
//                    return dynamic;
//                case Type.TypeKind.DimensionedTensorType:
//                    var tensor = type.AsTensorType();
//                    type.Dispose();
//                    return tensor;
//                default:
//                    return type;
//            }
//        }

//        [DllImport("LibTorchSharp")]
//        private static extern IntPtr THSJIT_forward(Module.HType module, IntPtr tensors, int length);

//        public TorchTensor Forward(params TorchTensor[] tensors)
//        {
//            var parray = new PinnedArray<IntPtr>();
//            IntPtr tensorRefs = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

//            return new TorchTensor(THSJIT_forward(handle, tensorRefs, parray.Array.Length));
//        }
//    }
//}
