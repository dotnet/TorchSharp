using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using TorchSharp.PInvoke;

namespace TorchSharp.Amp
{
    public class AMPManager : IDisposable
    {
        
        //TODO: Make Singleton THREADSAFE
        public class TensorConverter
        {
            //public torch.Tensor Tensor;
            public IntPtr PrevHandle;
            public IntPtr Handle;
            public torch.ScalarType Dtype;
            public torch.ScalarType FastDtype = torch.ScalarType.Float32;
            public TensorCalledIn Called, Status;
            public enum TensorCalledIn
            {
                OutSide,
                InsideEnter
            }

            public TensorConverter(IntPtr handle)
            {
                this.PrevHandle = handle;
                this.Handle = handle;
                this.Dtype = (torch.ScalarType)NativeMethods.THSTensor_type(handle);
                this.FastDtype = AutocastMode.GetInstance().GetFastType();
                
                Status = TensorConverter.TensorCalledIn.InsideEnter;
            }
            /*public TensorConverter(torch.Tensor tensor) : this(tensor.handle)
            {
                this.Tensor = tensor;
            }*/
        }

        public IList<TensorConverter> TensorsCasts = new List<TensorConverter>();
        public bool IsEnter = false;
        public bool IsDisposed = false;
        /*public UnorderedMap<IntPtr, torch.ScalarType> TensorPtrs= new UnorderedMap<IntPtr, torch.ScalarType>();
        public UnorderedMap<torch.Tensor, torch.ScalarType> TensorMap= new UnorderedMap<torch.Tensor, torch.ScalarType>();*/
        private AutocastMode autocastMode=null;
        public bool IsEnabled {
            get {
                if (autocastMode == null)
                    return false;
                return autocastMode.IsEnabled;
            }
        }

        private AMPManager(bool enabled)
        {
            if (!torch.cuda_is_available())
                return;
            autocastMode = AutocastMode.GetInstance(enabled);
        }

        private static AMPManager Instance;
        public static AMPManager GetInstance(bool enabled = false)
        {
            return Instance ??= new AMPManager(enabled);
        }

        private torch.ScalarType GetType(IntPtr handle)
        {
            return (torch.ScalarType)NativeMethods.THSTensor_type(handle);
        }

        public IntPtr AutoCast(IntPtr handle)
        {
            return ToIf(handle, AutocastMode.GetInstance().GetFastType());
        }

        public torch.Tensor AutoCast(torch.Tensor tensor)
        {
            return new torch.Tensor(AutoCast(tensor.Handle));
            //return tensor.to(AutocastMode.GetInstance().GetFastType());
        }
        public static IntPtr To(IntPtr ptr, torch.ScalarType type)
        {
            Debug.WriteLine($"{nameof(AMPManager)} Tensor converting from: {(torch.ScalarType)NativeMethods.THSTensor_type(ptr)} to: {type}");
            var res = NativeMethods.THSTensor_to_type(ptr, (sbyte)type);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return res;
        }
        public static IntPtr ToIf(IntPtr ptr, torch.ScalarType type)
        {
            if (!AMPManager.GetInstance().IsEnabled)
                return ptr;
            var res = NativeMethods.THSTensor_to_type(ptr, (sbyte)type);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return res;
        }
        private void Revert()
        {
            for (int i = 0; i < TensorsCasts.Count; i++) {
                var tc = TensorsCasts[i];
                //var tt = new torch.Tensor(tc.Handle);
                //var t = new torch.Tensor(tc.Handle) { handle = To(tc.Handle, tc.Dtype) };
                //var t = new torch.Tensor(tc.Handle).to(tc.Dtype);
                tc.Handle= To(tc.Handle, tc.Dtype);
                if (tc.Handle != tc.PrevHandle)
                    tc.PrevHandle = To(tc.PrevHandle, tc.Dtype);
            }
            //Cast Work very well but UNCASTING (if outscope, not working i dont know why...)
            //TensorsCasts.Clear();
        }
       

        private int ExistsHandle(IntPtr handle)
        {
            for (int i = 0; i < TensorsCasts.Count; i++)
                if (TensorsCasts[i].PrevHandle == handle || TensorsCasts[i].Handle == handle)
                    return i;
            return -1;
        }

        public IntPtr Work(IntPtr handle, IntPtr prev)
        {
            if (!this.IsEnabled)
                return handle;
            /*if (IsDisposed && !IsEnter) {
                Revert(); //Is for cleaned all
                return IntPtr.Zero;
            }*/
            var idx = ExistsHandle(handle);
            Console.WriteLine($"PTR: {handle}, PREV: {prev}, IDX: {idx}, {GetType(handle)}");
            if (idx == -1) {
                var tc = new TensorConverter(handle) { Called = IsEnter
                    ? TensorConverter.TensorCalledIn.InsideEnter
                    : TensorConverter.TensorCalledIn.OutSide
                };
                
                if (IsEnter)
                    tc.Handle = To(tc.Handle, tc.FastDtype);
                TensorsCasts.Add(tc);
                return tc.Handle;
            }
            var tcidx = TensorsCasts[idx];
            tcidx.Handle = handle;
            return tcidx.Handle;
            /*if (!IsEnter && IsDisposed) {
                if (tcidx.Called == TensorConverter.TensorCalledIn.OutSide) { //Is created outside so this can revert
                    //Is From Outside and is disposed, the tensor is created Outside so i will revert this
                    tcidx.PrevHandle = tcidx.Handle;
                    tcidx.Handle = To(tcidx.Handle, tcidx.Dtype);
                }
                return tcidx.Handle;
            }
            if (GetType(tcidx.Handle) == tcidx.FastDtype)
                return tcidx.Handle;

            if (IsEnter) {
                tcidx.PrevHandle = tcidx.Handle;
                tcidx.Handle = To(tcidx.Handle, tcidx.FastDtype);
            }
            return tcidx.Handle;*/
        }
        
        public IDisposable Enter()
        {
            if (!torch.cuda_is_available())
                return this;
            IsEnter = true;
            IsDisposed = false;
            autocastMode.SetEnabled(true, torch.CUDA);
            Debug.WriteLine($"{nameof(AMPManager)} Enter call");
            return this;
        }
        protected virtual void Dispose(bool disposing)
        {
            Debug.WriteLine($"{nameof(AMPManager)} Disposed call");
            IsDisposed = true;
            IsEnter = false;
            Revert();
            //Work(IntPtr.Zero, IntPtr.Zero);
            autocastMode.Dispose();
            //Revert();
            /*TensorPtrs.Dispose();
            TensorMap.Dispose();*/
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
        /*~AMPManager()
        {
            Dispose(false);
        }*/

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
