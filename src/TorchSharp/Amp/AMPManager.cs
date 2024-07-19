using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Google.Protobuf.WellKnownTypes;
using TorchSharp.PInvoke;
using TorchSharp.Utils;

namespace TorchSharp.Amp
{
    public class AMPManager : IDisposable
    {
        //TODO: Make Singleton THREADSAFE
        public UnorderedMap<IntPtr, torch.ScalarType> TensorPtrs;
        private readonly AutocastMode autocastMode = AutocastMode.GetInstance();

        private AMPManager() { }

        public bool IsEnabled => autocastMode.Enabled;
        private static AMPManager Instance;
        //bool disposedValue;

        public static AMPManager GetInstance()
        {
            return Instance ??= new AMPManager();
        }

        private void To(IntPtr ptr, torch.ScalarType type)
        {
            var res = NativeMethods.THSTensor_to_type(ptr, (sbyte)type);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
        }
        private void Revert()
        {
            using (var enumer = TensorPtrs.GetEnumerator())
                while (enumer.MoveNext())
                    To(enumer.Current.Key, enumer.Current.Value);
            TensorPtrs.Clear(); //Or should use Stack for POP?? May better performance and better ram usage
        }

        public void Add(IntPtr ptr)
        {
            if (!autocastMode.Enabled) {
                
                if (TensorPtrs.ContainsKey(ptr))
                    To(ptr, TensorPtrs[ptr]);
                return;
            }

            TensorPtrs[ptr] = (torch.ScalarType)NativeMethods.THSTensor_type(ptr);
            To(ptr, autocastMode.GetFastType()); //TODO: Set scalar autocast
        }

        public IDisposable Enter()
        {
            return null;
        }
        protected virtual void Dispose(bool disposing)
        {
            Revert();
            autocastMode.Dispose();
            /*if (!disposedValue) {
                if (disposing) {
                    
                    
                    // TODO: dispose managed state (managed objects)
                }
                
                // TODO: free unmanaged resources (unmanaged objects) and override finalizer
                // TODO: set large fields to null
                disposedValue = true;
            }*/
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        ~AMPManager()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
