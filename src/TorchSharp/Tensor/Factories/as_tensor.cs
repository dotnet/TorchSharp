// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System.Collections.Generic;
using System.Linq;

#nullable enable
namespace TorchSharp
{
    public static partial class torch
    {
        // as_tensor()

        public static Tensor as_tensor(Tensor data, ScalarType? dtype = null, Device? device = null)
        {
            if (dtype != null && device != null && (data.dtype != dtype || data.device != device)) {
                return data.to(dtype.Value, device).requires_grad_(data.requires_grad);
            } else if (dtype != null && data.dtype != dtype) {
                return data.to(dtype.Value).requires_grad_(data.requires_grad);
            } else if (device != null && data.device != device) {
                return data.to(device).requires_grad_(data.requires_grad);
            } else {
                return data.alias();
            }
        }

        public static Tensor as_tensor(IList<bool> rawArray, ScalarType? dtype = null, Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(bool[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<byte> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(byte[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<sbyte> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(sbyte[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<short> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(short[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<int> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(int[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<long> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(long[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<float> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(float[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<double> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(double[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<(float, float)> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor((float, float)[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<System.Numerics.Complex> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(System.Numerics.Complex[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<BFloat16> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(BFloat16[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }
    }
}
