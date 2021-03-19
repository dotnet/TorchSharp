// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Module : IDisposable
    {
        /// <summary>
        ///    Class wrapping PyTorch's module object reference.
        /// </summary>
        internal sealed class HType : SafeHandle
        {
            public HType (IntPtr preexistingHandle, bool ownsHandle) : base (IntPtr.Zero, ownsHandle)
            {
                SetHandle (preexistingHandle);
            }

            public override bool IsInvalid => handle == IntPtr.Zero;

            // This is just for marshalling
            internal HType () : base (IntPtr.Zero, true)
            {
            }

            [DllImport ("LibTorchSharp")]
            private static extern void THSNN_Module_dispose (HType handle);

            protected override bool ReleaseHandle ()
            {
                THSNN_Module_dispose (this);
                return true;
            }

            protected override void Dispose (bool disposing)
            {
                if (disposing) {
                    ReleaseHandle ();
                }
            }
        }

        internal HType handle;

        /// Stores the AnyModule corresponding to this module.
        internal BoxedModule boxedModule;

        internal BoxedModule BoxedModule
        {
            get
            {
                if (boxedModule == null)
                    throw new InvalidOperationException ("A Sequential or Loaded module may not be added to a Sequential");
                return boxedModule;
            }
        }

        internal Module (IntPtr handle, IntPtr? boxedHandle, bool ownsHandle = true)
        {
            this.handle = new HType (handle, ownsHandle);
            this.boxedModule = boxedHandle.HasValue ? new BoxedModule(boxedHandle.Value) : null;
            //Debug.Assert (!this.handle.IsInvalid);
            //Debug.Assert (!this.boxedHandle.IsInvalid);
        }

        ~Module ()
        {
            Dispose (false);
        }

        /// <summary>
        ///   Releases the storage.
        /// </summary>
        public void Dispose ()
        {
            Dispose (true);
            GC.SuppressFinalize (this);
        }

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        protected void Dispose (bool disposing)
        {
            if (disposing) {
                handle.Dispose ();
                handle.SetHandleAsInvalid ();
            }
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSNN_Module_load([MarshalAs(UnmanagedType.LPStr)] string location);

        public static Module Load(String location)
        {
            var handle = THSNN_Module_load (location);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Module (handle, IntPtr.Zero);
        }

        [DllImport ("LibTorchSharp")]
        extern static void THSNN_Module_save (HType handle, [MarshalAs(UnmanagedType.LPStr)] string location);

        public virtual void Save (String modelPath)
        {
            THSNN_Module_save (handle, modelPath);
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSNN_Module_train(HType module);

        public virtual void Train ()
        {
            THSNN_Module_train (handle);
            Torch.CheckForErrors ();
        }

        [DllImport ("LibTorchSharp")]
        private static extern void THSNN_Module_eval (HType module);

        public virtual void Eval ()
        {
            THSNN_Module_eval (handle);
            Torch.CheckForErrors ();
        }

        [DllImport ("LibTorchSharp")]
        private static extern bool THSNN_Module_is_training (HType module);

        public bool IsTraining ()
        {
            var res = THSNN_Module_is_training (handle);
            Torch.CheckForErrors ();
            return res;
        }

        [DllImport ("LibTorchSharp")]
        private static extern void THSNN_Module_zero_grad (HType module);

        public virtual void ZeroGrad ()
        {
            THSNN_Module_zero_grad (handle);
            Torch.CheckForErrors ();
        }

        [DllImport ("LibTorchSharp")]
        private static extern void THSNN_Module_get_named_parameters (HType module, AllocatePinnedArray allocator1, AllocatePinnedArray allocator2);

        public virtual (string name, TorchTensor parameter)[] NamedParameters ()
        {
            IntPtr[] ptrArray;
            IntPtr[] strArray;

            using (var pa = new PinnedArray<IntPtr> ())
            using (var sa = new PinnedArray<IntPtr> ()) {
                THSNN_Module_get_named_parameters (handle, pa.CreateArray, sa.CreateArray);
                Torch.CheckForErrors ();
                ptrArray = pa.Array;
                strArray = sa.Array;
            }
            return ptrArray.Select((x, i) => (Marshal.PtrToStringAnsi(strArray[i]), new TorchTensor(x))).ToArray();

    }

        [DllImport ("LibTorchSharp")]
        private static extern void THSNN_Module_get_parameters (HType module, AllocatePinnedArray allocator);

        public virtual TorchTensor[] GetParameters ()
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr> ()) {
                AllocatePinnedArray allocator = pa.CreateArray;
                THSNN_Module_get_parameters (handle, allocator);
                Torch.CheckForErrors ();
                ptrArray = pa.Array;
            }
            return ptrArray.Select(x => new TorchTensor (x)).ToArray();
        }

        [DllImport ("LibTorchSharp")]
        static extern bool THSNN_Module_has_parameter (HType module, [MarshalAs(UnmanagedType.LPStr)]string name);

        public bool HasParameter (string name)
        {
            var res = THSNN_Module_has_parameter (handle, name);
            Torch.CheckForErrors ();
            return res;
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Module_get_parameter (HType module, [MarshalAs(UnmanagedType.LPStr)] string name);

        public TorchTensor GetParameter (string name)
        {
            var parameter = THSNN_Module_get_parameter (handle, name);
            Torch.CheckForErrors ();

            if (parameter == IntPtr.Zero) {
                throw new ArgumentNullException ("Linear module without bias term.");
            }

            return new TorchTensor (parameter);
        }
        [DllImport ("LibTorchSharp")]
        private static extern void THSNN_Module_register_module (HType module, string name, HType submodule);

        public virtual void RegisterModule (string name, Module submodule)
        {
            THSNN_Module_register_module (handle, name, submodule.handle);
            Torch.CheckForErrors ();
        }

        [DllImport ("LibTorchSharp")]
        private static extern long THSNN_Module_children_size (HType module);

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Module_child (HType module, int index);

        /// Get the sub-modules of this module. The Module objects won't have the correct .NET types
        /// so this is not made public.
        internal virtual Module[] GetModulesInternal ()
        {
            var numModules = THSNN_Module_children_size (handle);
            Torch.CheckForErrors ();
            Module[] result = new Module[numModules];

            for (int i = 0; i < numModules; i++) {
                var childHandle = THSNN_Module_child (handle, i);
                Torch.CheckForErrors ();
                result[i] = new Module (childHandle, null, ownsHandle: false);
            }

            return result;
        }

        [DllImport ("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.LPStr)]
        private static extern string THSNN_Module_name (HType module);

        public virtual string GetName ()
        {
            var res = THSNN_Module_name (handle);
            Torch.CheckForErrors ();
            return res;
        }
    }

    internal class BoxedModule : IDisposable
    {
        internal sealed class HType : SafeHandle
        {
            public HType (IntPtr preexistingHandle, bool ownsHandle) : base (IntPtr.Zero, ownsHandle)
            {
                SetHandle (preexistingHandle);
            }

            public override bool IsInvalid => handle == IntPtr.Zero;

            // This is just for marshalling
            internal HType () : base (IntPtr.Zero, true)
            {
            }

            [DllImport ("LibTorchSharp")]
            private static extern void THSNN_AnyModule_dispose (HType handle);

            protected override bool ReleaseHandle ()
            {
                THSNN_AnyModule_dispose (this);
                return true;
            }

            protected override void Dispose (bool disposing)
            {
                if (disposing) {
                    ReleaseHandle ();
                }
            }
        }

        internal HType handle;

        internal BoxedModule (IntPtr handle)
        {
            this.handle = new HType (handle, true);
        }

        ~BoxedModule ()
        {
            Dispose (false);
        }

        /// <summary>
        ///   Releases the storage.
        /// </summary>
        public void Dispose ()
        {
            Dispose (true);
            GC.SuppressFinalize (this);
        }

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        protected void Dispose (bool disposing)
        {
            if (disposing) {
                handle.Dispose ();
                handle.SetHandleAsInvalid ();
            }
        }
    }

    public abstract class CustomModule : Module
    {
        private delegate IntPtr ForwardFunctionC (IntPtr tensor);

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_custom_module([MarshalAs(UnmanagedType.LPStr)] string name,
            IntPtr names, IntPtr parameters, IntPtr require_grad,
            int length, ForwardFunctionC forward, out IntPtr pBoxedModule);

        protected CustomModule (string name, params Parameter[] parameters) : base (IntPtr.Zero, IntPtr.Zero)
        {
            var names = parameters.Select (p => Marshal.StringToHGlobalAnsi (p.Name)).ToArray ();
            var @params = parameters.Select (p => p.Tensor.Handle).ToArray ();
            var withGrads = parameters.Select (p => p.WithGrad).ToArray ();

            var namesPinned = new PinnedArray<IntPtr> ();
            var paramsPinned = new PinnedArray<IntPtr> ();
            var wGradPinned = new PinnedArray<bool> ();

            var nparray = namesPinned.CreateArray (names);
            var pparray = paramsPinned.CreateArray (@params);
            var gparray = wGradPinned.CreateArray (withGrads);

            ForwardFunctionC forwardNative = t => (forward (new TorchTensor (t)).Handle);
            var res = THSNN_custom_module (name, nparray, pparray, gparray, names.Length, forwardNative, out var boxedHandle);
            Torch.CheckForErrors ();
            this.handle = new HType (res, true);
            this.forwardNative = forwardNative;
            this.boxedModule = new BoxedModule(boxedHandle);
        }

        /// Keeps the callback delegate alive
        private ForwardFunctionC forwardNative;

        abstract public TorchTensor forward (TorchTensor t);
    }
}
