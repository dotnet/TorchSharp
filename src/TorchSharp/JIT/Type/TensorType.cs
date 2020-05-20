//// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
//using System;
//using System.Runtime.InteropServices;

//namespace TorchSharp.JIT
//{
//    public sealed class TensorType : Type
//    {
//        internal TensorType(IntPtr handle) : base(handle)
//        {
//            this.handle = new HType(handle, true);
//        }

//        internal TensorType(Type type) : base()
//        {
//            handle = type.handle;
//            type.handle = new HType(IntPtr.Zero, true);
//            type.Dispose();
//        }

//        [DllImport("LibTorchSharp")]
//        private static extern short THSJIT_getScalarFromDimensionedTensorType(HType handle);

//        public Tensor.ScalarType GetScalarType()
//        {
//            return (Tensor.ScalarType)THSJIT_getScalarFromDimensionedTensorType(handle);
//        }

//        [DllImport("LibTorchSharp")]
//        private static extern int THSJIT_getDimensionedTensorTypeDimensions(HType handle);

//        public int GetDimensions()
//        {
//            return THSJIT_getDimensionedTensorTypeDimensions(handle);
//        }

//        [DllImport("LibTorchSharp")]
//        private static extern string THSJIT_getDimensionedTensorDevice(HType handle);

//        public string GetDevice()
//        {
//            return THSJIT_getDimensionedTensorDevice(handle);
//        }
//    }
//}
