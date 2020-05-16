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

        internal Module (IntPtr handle)
        {
            this.handle = new HType (handle, true);
            Debug.Assert (!this.handle.IsInvalid);
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_new_module (IntPtr names, IntPtr parameters, IntPtr with_grad, int length);

        protected Module (params Parameter[] parameters)
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

            var res = THSNN_new_module (nparray, pparray, gparray, names.Length);
            Torch.CheckForErrors ();
            handle = new HType (res, true);
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
        extern static IntPtr THSNN_Module_load(string location);

        public static Module Load(String location)
        {
            var handle = THSNN_Module_load (location);
            Torch.CheckForErrors ();
            return new Module (handle);
        }

        [DllImport ("LibTorchSharp")]
        extern static void THSNN_Module_save (HType handle, string location);

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

        public virtual IEnumerable<(string name, TorchTensor parameter)> NamedParameters ()
        {
            //// If module has no children, fetch the paramters from pytorch
            //if (Modules.Any ()) {
            //    IEnumerable<(string name, TorchTensor parameter)> result = Enumerable.Empty<(string name, TorchTensor parameter)> ();

            //    foreach (var module in Modules) {
            //        result = result.Concat (module.NamedParameters ());
            //    }

            //    return result;
            //}

            IntPtr[] ptrArray;
            IntPtr[] strArray;

            using (var pa = new PinnedArray<IntPtr> ())
            using (var sa = new PinnedArray<IntPtr> ()) {
                THSNN_Module_get_named_parameters (handle, pa.CreateArray, sa.CreateArray);
                Torch.CheckForErrors ();
                ptrArray = pa.Array;
                strArray = sa.Array;
            }
            return strArray.Select (s => Marshal.PtrToStringAnsi (s)).Zip (ptrArray.Select (x => new TorchTensor (x)), (x, y) => (x, y));
        }

        [DllImport ("LibTorchSharp")]
        private static extern void THSNN_Module_get_parameters (HType module, AllocatePinnedArray allocator);

        public virtual IEnumerable<TorchTensor> Parameters ()
        {
            //// If module has no children, fetch the paramters from pytorch, otherwise iterate over the params of the child modules
            //if (Modules.Any ()) {
            //    IEnumerable<TorchTensor> result = Enumerable.Empty<TorchTensor> ();

            //    foreach (var module in Modules) {
            //        result = result.Concat (module.Parameters ());
            //    }

            //    return result;
            //}

            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr> ()) {
                THSNN_Module_get_parameters (handle, pa.CreateArray);
                Torch.CheckForErrors ();
                ptrArray = pa.Array;
            }
            return ptrArray.Select (x => new TorchTensor (x));
        }

        [DllImport ("LibTorchSharp")]
        private static extern bool THSNN_Module_has_parameter (HType module, string name);

        public bool HasParameter (string name)
        {
            var res = THSNN_Module_has_parameter (handle, name);
            Torch.CheckForErrors ();
            return res;
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Module_get_parameter (HType module, string name);

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
        private static extern IntPtr THSNN_Module_register_module (HType module, string name, HType submodule);

        public virtual Module RegisterModule (string name, Module submodule)
        {
            var res= THSNN_Module_register_module (handle, name, submodule.handle);
            Torch.CheckForErrors ();
            return new Module (res);
        }

        [DllImport ("LibTorchSharp")]
        private static extern long THSNN_Module_children_size (HType module);

        [DllImport ("LibTorchSharp")]
        private static extern string THSNN_Module_child_name (HType module, int index);

        public virtual IEnumerable<string> GetModules ()
        {
            var numModules = THSNN_Module_children_size (handle);
            Torch.CheckForErrors ();
            string[] result = new string[numModules];

            for (int i = 0; i < numModules; i++) {
                result[i] = THSNN_Module_child_name (handle, i);
                Torch.CheckForErrors ();
            }

            return result;
        }

        [DllImport ("LibTorchSharp")]
        private static extern string THSNN_Module_name (HType module);

        public virtual string GetName ()
        {
            var res = THSNN_Module_name (handle);
            Torch.CheckForErrors ();
            return res;
        }
    }

}
