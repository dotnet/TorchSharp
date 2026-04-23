using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using TorchSharp.PInvoke;


namespace TorchSharp.BitsAndBytes
{
    //BASED ON: https://github.com/LittleLittleCloud/TorchSharp.BitsAndBytes
    public class BitsAndByteUtils
    {
        /// <summary>
        /// [methodname, quantized type, scalar type] -> [MethodInfo]
        /// </summary>
        static readonly Dictionary<string, MethodInfo> bitsandbyte_methods_natives = new Dictionary<string, MethodInfo>();
        public static void Initialize()
        {
            var methods = typeof(BitsAndBytesNatives).GetMethods(BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly)
                .Where(x=>x.Name.StartsWith("cquantize") ||
                          x.Name.StartsWith("cdequantize") ||
                          x.Name.StartsWith("cgemm_4bit"));
            foreach (var method in methods) {
                bitsandbyte_methods_natives.Add(method.Name, method);
            }
        }


        private static string GetScalarTypeString(torch.ScalarType st)
        {
            if (st == torch.ScalarType.Float32)
                return "fp32";
            if (st == torch.ScalarType.BFloat16)
                return "bf16";
            return "fp16";
        }
        private static readonly Lazy<Dictionary<(string, string, int), torch.Tensor>> _4bitTypeCache = new Lazy<Dictionary<(string, string, int), torch.Tensor>>();
        public static (
                torch.Tensor quantizedTensor,
                torch.Tensor absMax,
                int blockSize,
                int n
                )
                Quantize4Bit(
                torch.Tensor tensor, // input tensor
                string quantizedDType = "fp4", // quantized data type, must be one of "fp4", "nf4"
                int blockSize = 64 // block size
                )
        {
            var n = (int)tensor.numel();
            var blocks = (int)Math.Ceiling((double)n / blockSize);
            var absMax = torch.zeros(new long[]{blocks}, dtype: torch.float32).cuda();
            var mod = 2;
            var quantizedTensor = torch.zeros(new long[]{n+1, mod, 1}, dtype: torch.ScalarType.Byte).cuda();
            if(bitsandbyte_methods_natives.Count == 0)
                Initialize();
            if(!bitsandbyte_methods_natives.TryGetValue($"cquantize_blockwise_{GetScalarTypeString(tensor.dtype)}_{quantizedDType}", out var m))
                throw new NotImplementedException();

            m.Invoke(
                null,
                new object[]{
                    IntPtr.Zero,
                    NativeMethods.THSStorage_data_ptr(tensor.Handle),
                    NativeMethods.THSStorage_data_ptr(absMax.Handle),
                    NativeMethods.THSStorage_data_ptr(quantizedTensor.Handle),
                    blockSize,
                    n
                }
            );
            return (quantizedTensor, absMax, blockSize, n);
        }

        public static torch.Tensor Dequantize4Bit(
            torch.Tensor tensor, // quantized tensor
            torch.Tensor absMax, // absMax tensor
            torch.ScalarType originalDType, // original data type
            string quantizedDType, // quantized data type, must be one of "fp4", "nf4"
            int n,
            long[] originalShape,
            int blockSize = 64, // block size
            torch.ScalarType quantStorageDType = torch.ScalarType.Byte // quantized storage data type
            )
        {

            var dequantizedTensor = torch.zeros(originalShape, dtype: originalDType).cuda();
            if (bitsandbyte_methods_natives.Count == 0)
                Initialize();
            if (!bitsandbyte_methods_natives.TryGetValue($"cdequantize_blockwise_{GetScalarTypeString(originalDType)}_{quantizedDType}", out var m))
                throw new NotImplementedException();

            m.Invoke(
                null,
                new object[]{
                    IntPtr.Zero,
                    NativeMethods.THSStorage_data_ptr(tensor.Handle),
                    NativeMethods.THSStorage_data_ptr(absMax.Handle),
                    NativeMethods.THSStorage_data_ptr(dequantizedTensor.Handle),
                    blockSize,
                    n,
                    IntPtr.Zero
                }
            );
            return dequantizedTensor;
        }

        public static torch.Tensor Get4BitType(string typename, string device = "cuda", int blocksize = 64)
        {
            if (_4bitTypeCache.Value.TryGetValue((typename, device, blocksize), out var cachedTensor)) {
                return cachedTensor;
            }

            float[] data = null;

            if (typename == "nf4") {
                // Implements the NF4 data type.
                // Constructs a quantization data type where each bin has equal area under a standard normal distribution N(0, 1) that
                // is normalized into the range [-1, 1].
                data = new float[] {
                    -1.0f,
                    -0.6961928f,
                    -0.5250731f,
                    -0.3949175f,
                    -0.2844414f,
                    -0.1847734f,
                    -0.09105004f,
                    0.0f,
                    0.0795803f,
                    0.1609302f,
                    0.2461123f,
                    0.3379152f,
                    0.4407098f,
                    0.562617f,
                    0.7229568f,
                    1.0f
                };
            }
            else if (typename == "fp4") {
                data = new float[]
                {
                0.0f, 0.0625f, 8.0f, 12.0f, 4.0f, 6.0f, 2.0f, 3.0f,
                -0.0f, -0.0625f, -8.0f, -12.0f, -4.0f, -6.0f, -2.0f, -3.0f
                };
            }
            else if (typename == "int4") {
                data = new float[] { 7, 6, 5, 4, 3, 2, 1, 0, -0, -1, -2, -3, -4, -5, -6, -7 };
            }
            else if (typename == "af4") {
                if (blocksize == 64) {
                    data = new float[] {
                        -1.0f, -0.69441008f, -0.51243739f, -0.3736951f, -0.25607552f, -0.14982478f, -0.04934812f, 0.0f,
                        0.04273164f, 0.12934483f, 0.21961274f, 0.31675666f, 0.42563882f, 0.55496234f, 0.72424863f, 1.0f
                    };
                    Array.Reverse(data);
                } else {
                    throw new NotImplementedException("4-bit AbnormalFloats currently only support blocksize 64.");
                }
            }

            if (data == null) {
                throw new NotImplementedException($"Typename {typename} not supported");
            }

            var tensor = torch.tensor(data, device: device);
            tensor.div_(tensor.abs().max());

            if (tensor.numel() != 16) {
                throw new Exception("Tensor does not have 16 elements.");
            }

            _4bitTypeCache.Value[(typename, device, blocksize)] = tensor;
            tensor.DetachFromDisposeScope();
            return tensor;
        }

        public static torch.Tensor Gemv4Bit(
            torch.Tensor input,
            torch.Tensor quantizedWeight,
            long[] originalWeightShape,
            torch.Tensor absMax,
            int blockSize,
            string quantizedDType) // quantized data type, must be one of "fp4", "nf4"
        {
            var inputShape = input.IntShape();
            if (input.numel() != inputShape[^1]) {
                throw new ArgumentException("'Dimensions of A are invalid. Must be a vector with the leading dimensions of \"1\", e.g. [1, 1, 2048]'");
            }
            var batch = inputShape[0];
            var inputDType = input.dtype;
            var m = (int)originalWeightShape[0];
            var k = (int)originalWeightShape[1];
            var lda = (int)originalWeightShape[0];
            var ldc = (int)originalWeightShape[0];
            var ldb = (inputShape[^1] + 1) / 2;
            torch.Tensor output;
            if (inputShape.Length == 3) {
                output = torch.zeros(new long[] { batch, inputShape[1], originalWeightShape[0]}, dtype: inputDType).cuda();
            } else {
                output = torch.zeros(new long[]{batch, originalWeightShape[0]}, dtype: inputDType).cuda();
            }

            // quantize weight
            var code = Get4BitType(quantizedDType, "cuda", blockSize);

            if (bitsandbyte_methods_natives.Count == 0)
                Initialize();

            if (!bitsandbyte_methods_natives.TryGetValue($"cgemm_4bit_inference_naive_{GetScalarTypeString(inputDType)}", out var mt))
                throw new NotImplementedException();

            mt.Invoke(null, new object[] {
                m,batch,k,input.GetDataPtr(), quantizedWeight.T.GetDataPtr(),
                absMax.GetDataPtr(),
                code.GetDataPtr(),
                output.GetDataPtr(),
                lda,
                ldb,
                ldc,
                blockSize,
                IntPtr.Zero
            });
            return output;
        }


        public static torch.Tensor CreateDynamicMap(bool signed = true, int maxExponentBits = 7, int totalBits = 8)
        {
            var data = new List<float>();
            int nonSignBits = totalBits - (signed ? 1 : 0);
            int additionalItems = (int)Math.Pow(2, nonSignBits - maxExponentBits) - 1;

            for (int i = 0; i < maxExponentBits; i++) {
                /*int fractionItems = signed
                    ? (int)Math.Pow(2, i + nonSignBits - maxExponentBits) + 1
                    : (int)Math.Pow(2, i + nonSignBits - maxExponentBits + 1) + 1;*/

                int fractionItems = (int)Math.Pow(2, i + nonSignBits - maxExponentBits + (signed ? 1 : 0)) + 1;

                var boundaries = torch.linspace(0.1, 1, fractionItems);
                var means = (boundaries[..^1] + boundaries[1..]) / 2.0;
                data.AddRange((torch.pow(10f, i - (maxExponentBits - 1)) * means).data<float>().ToArray());

                if (signed) {
                    data.AddRange((-(torch.pow(10f, (-(maxExponentBits - 1) + i)) * means)).data<float>().ToArray());
                }
            }

            if (additionalItems > 0) {
                var boundaries = torch.linspace(0.1, 1, additionalItems + 1);
                var means = (boundaries[..^1] + boundaries[1..]) / 2.0;
                data.AddRange((torch.pow(10f, -(maxExponentBits - 1) + maxExponentBits - 1) * means).data<float>().ToArray());

                if (signed) {
                    data.AddRange((-(torch.pow(10f, -(maxExponentBits - 1) + maxExponentBits - 1) * means)).data<float>().ToArray());
                }
            }

            data.AddRange(new float[] { 0, 1.0f });

            if (data.Count != (int)Math.Pow(2, totalBits)) {
                int gap = 256 - data.Count;
                for (int i = 0; i < gap; i++) {
                    data.Add(0);
                }
            }

            data.Sort();
            return torch.tensor(data.ToArray());
        }

        public static int[] CheckMatmul(torch.Tensor A, torch.Tensor B, bool transposed_A, bool transposed_B, torch.ScalarType expectedType = torch.ScalarType.Int8)
        {
            if (A.dtype != expectedType || B.dtype != expectedType) {
                throw new ArgumentException($"Expected {expectedType} input tensors A and B, but got {A.dtype} and {B.dtype}");
            }

            var sA = A.IntShape();
            var sB = B.IntShape();
            var tA = transposed_A;
            var tB = transposed_B;

            bool correct = true;

            if (sA.Length == 2 && sB.Length == 2) {
                if (!tA && !tB && A.shape[1] != B.shape[0]) {
                    correct = false;
                } else if (tA && !tB && A.shape[0] != B.shape[0]) {
                    correct = false;
                } else if (tA && tB && A.shape[0] != B.shape[1]) {
                    correct = false;
                } else if (!tA && tB && A.shape[1] != B.shape[1]) {
                    correct = false;
                }
            } else if (sA.Length == 3 && sB.Length == 2) {
                if (!tA && !tB && A.shape[2] != B.shape[0]) {
                    correct = false;
                } else if (tA && !tB && A.shape[1] != B.shape[0]) {
                    correct = false;
                } else if (tA && tB && A.shape[1] != B.shape[1]) {
                    correct = false;
                } else if (!tA && tB && A.shape[2] != B.shape[1]) {
                    correct = false;
                }
            } else if (sA.Length == 3 && sB.Length == 3) {
                if (!tA && !tB && A.shape[2] != B.shape[1]) {
                    correct = false;
                } else if (tA && !tB && A.shape[1] != B.shape[1]) {
                    correct = false;
                } else if (tA && tB && A.shape[1] != B.shape[2]) {
                    correct = false;
                } else if (!tA && tB && A.shape[2] != B.shape[2]) {
                    correct = false;
                }
            }

            int[] outShape = null;

            if (sA.Length == 2 && sB.Length == 2) {
                if (!tA && !tB) {
                    outShape = new int[] { sA[0], sB[1] };
                } else if (tA && tB) {
                    outShape = new int[] { sA[1], sB[0] };
                } else if (tA && !tB) {
                    outShape = new int[] { sA[1], sB[1] };
                } else if (!tA && tB) {
                    outShape = new int[] { sA[0], sB[0] };
                }
            } else if (sA.Length == 3 && sB.Length == 2) {
                if (!tA && !tB) {
                    outShape = new int[] { sA[0], sA[1], sB[1] };
                } else if (tA && tB) {
                    outShape = new int[] { sA[0], sA[2], sB[0] };
                } else if (tA && !tB) {
                    outShape = new int[] { sA[0], sA[2], sB[1] };
                } else if (!tA && tB) {
                    outShape = new int[]{sA[0], sA[1], sB[0]};
                }
            } else if (sA.Length == 3 && sB.Length == 3) {
                if (!tA && !tB) {
                    outShape = new int[] { sA[0], sA[1], sB[2] };
                } else if (tA && tB) {
                    outShape = new int[] { sA[0], sA[2], sB[1] };
                } else if (tA && !tB) {
                    outShape = new int[] { sA[0], sA[2], sB[2] };
                } else if (!tA && tB) {
                    outShape = new int[] { sA[0], sA[1], sB[1] };
                }
            }

            if (!correct) {
                throw new ArgumentException(
                    $"Tensor dimensions incorrect for matrix multiplication: A x B: {sA.ToArray()} x {sB.ToArray()} with transpose for A x B: {tA} x {tB}."
                );
            }

            return outShape;
        }
    }
}
