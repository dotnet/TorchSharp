using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace TorchSharp.BitsAndBytes
{
    //BASED ON: https://github.com/LittleLittleCloud/TorchSharp.BitsAndBytes
    public static class BitsAndBytesNatives
    {
        private const string DllName = "libbitsandbytes";
        
        [DllImport(DllName)]
        public static extern void cdequantize_blockwise_fp32_fp4(
            IntPtr code,        // float*
            IntPtr A,          // float*
            IntPtr absmax,     // float*
            IntPtr output,     // unsigned char*
            int blocksize,
            int n,             // total size
            IntPtr stream);

        [DllImport(DllName)]
        public static extern void cdequantize_blockwise_fp32_nf4(
            IntPtr code,        // float*
            IntPtr A,          // float*
            IntPtr absmax,     // float*
            IntPtr output,     // unsigned char*
            int blocksize,
            int n,             // total size
            IntPtr stream);

        [DllImport(DllName)]
        public static extern void cdequantize_blockwise_fp16_fp4(
            IntPtr code,        // float*
            IntPtr A,          // float*
            IntPtr absmax,     // float*
            IntPtr output,     // unsigned char*
            int blocksize,
            int n,             // total size
            IntPtr stream);

        [DllImport(DllName)]
        public static extern void cdequantize_blockwise_fp16_nf4(
            IntPtr code,        // float*
            IntPtr A,          // float*
            IntPtr absmax,     // float*
            IntPtr output,     // unsigned char*
            int blocksize,
            int n,             // total size
            IntPtr stream);

        [DllImport(DllName)]
        public static extern void cdequantize_blockwise_bf16_fp4(
            IntPtr code,        // float*
            IntPtr A,          // float*
            IntPtr absmax,     // float*
            IntPtr output,     // unsigned char*
            int blocksize,
            int n,             // total size
            IntPtr stream);

        [DllImport(DllName)]
        public static extern void cdequantize_blockwise_bf16_nf4(
            IntPtr code,        // float*
            IntPtr A,          // float*
            IntPtr absmax,     // float*
            IntPtr output,     // unsigned char*
            int blocksize,
            int n,             // total size
            IntPtr stream
        );

        [DllImport(DllName)]
        public static extern void cquantize_blockwise_fp32_fp4(
            IntPtr code,        // float*
            IntPtr A,          // float*
            IntPtr absmax,     // float*
            IntPtr output,     // unsigned char*
            int blocksize,
            int n             // total size
        );

        [DllImport(DllName)]
        public static extern void cquantize_blockwise_fp32_nf4(
            IntPtr code,        // float*
            IntPtr A,          // float*
            IntPtr absmax,     // float*
            IntPtr output,     // unsigned char*
            int blocksize,
            int n             // total size
         );

        [DllImport(DllName)]
        public static extern void cquantize_blockwise_fp32(
            IntPtr code,        // float*
            IntPtr A,          // float*
            IntPtr absmax,     // float*
            IntPtr output,     // unsigned char*
            int blocksize,
            int n             // total size
         );

        [DllImport(DllName)]
        public static extern void cquantize_blockwise_fp16_fp4(
            IntPtr code,        // float*
            IntPtr A,          // float*
            IntPtr absmax,     // float*
            IntPtr output,     // unsigned char*
            int blocksize,
            int n             // total size
         );

        [DllImport(DllName)]
        public static extern void cquantize_blockwise_fp16_nf4(
            IntPtr code,        // float*
            IntPtr A,          // float*
            IntPtr absmax,     // float*
            IntPtr output,     // unsigned char*
            int blocksize,
            int n             // total size
         );

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void cquantize_blockwise_bf16_fp4(
            IntPtr code,        // float*
            IntPtr A,          // __nv_bfloat16*
            IntPtr absmax,     // float*
            IntPtr output,     // unsigned char*
            int blocksize,
            int n             // total size
        );

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void cquantize_blockwise_bf16_nf4(
            IntPtr code,        // float*
            IntPtr A,          // __nv_bfloat16*
            IntPtr absmax,     // float*
            IntPtr output,     // unsigned char*
            int blocksize,
            int n             // total size
        );

        [DllImport(DllName)]
        public static extern void cgemm_4bit_inference_naive_fp16(
            int m,
            int n,
            int k,
            IntPtr A,          // half*
            IntPtr B,          // unsigned char*
            IntPtr absmax,     // float*
            IntPtr datatype,   // float*
            IntPtr output,     // half*
            int lda,
            int ldb,
            int ldc,
            int blocksize,
            IntPtr stream      // cudaStream_t
        );

        [DllImport(DllName)]
        public static extern void cgemm_4bit_inference_naive_fp32(
            int m,
            int n,
            int k,
            IntPtr A,          // half*
            IntPtr B,          // unsigned char*
            IntPtr absmax,     // float*
            IntPtr datatype,   // float*
            IntPtr output,     // half*
            int lda,
            int ldb,
            int ldc,
            int blocksize,
            IntPtr stream      // cudaStream_t
        );

        [DllImport(DllName)]
        public static extern void cgemm_4bit_inference_naive_bf16(
            int m,
            int n,
            int k,
            IntPtr A,          // half*
            IntPtr B,          // unsigned char*
            IntPtr absmax,     // float*
            IntPtr datatype,   // float*
            IntPtr output,     // half*
            int lda,
            int ldb,
            int ldc,
            int blocksize,
            IntPtr stream      // cudaStream_t
        );

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void dequantize(
            IntPtr output,  // float* 
            IntPtr input,   // byte*
            IntPtr scale,   // float*
            int size,
            IntPtr stream   // cudaStream_t
        );

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void cigemm(
            IntPtr context,
            bool transposeA,
            bool transposeB,
            int m,
            int n,
            int k,
            IntPtr A, // input
            IntPtr B, // weight
            IntPtr C, // output
            int lda,
            int ldb,
            int ldc);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr get_context();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr get_cusparse();
    }
}
