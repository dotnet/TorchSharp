using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace TorchSharp.PInvoke
{
    internal static partial class NativeMethods
    {
        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.Struct)]
        internal static extern BFloat16 THSBFloat16_ctor(float value);

        [DllImport("LibTorchSharp")]
        internal static extern float THSBFloat16_op_float(BFloat16 bf16);
        [DllImport("LibTorchSharp")]
        internal static extern BFloat16 THSBFloat16_op_add(BFloat16 a, BFloat16 b);
        [DllImport("LibTorchSharp")]
        internal static extern BFloat16 THSBFloat16_op_sub(BFloat16 a, BFloat16 b);
        [DllImport("LibTorchSharp")]
        internal static extern BFloat16 THSBFloat16_op_mul(BFloat16 a, BFloat16 b);
        [DllImport("LibTorchSharp")]
        internal static extern BFloat16 THSBFloat16_op_div(BFloat16 a, BFloat16 b);
        [DllImport("LibTorchSharp")]
        internal static extern float THSBFloat16_op_add_float(BFloat16 a, float b);
        [DllImport("LibTorchSharp")]
        internal static extern float THSBFloat16_op_sub_float(BFloat16 a, float b);
        [DllImport("LibTorchSharp")]
        internal static extern float THSBFloat16_op_mul_float(BFloat16 a, float b);
        [DllImport("LibTorchSharp")]
        internal static extern float THSBFloat16_op_div_float(BFloat16 a, float b);
        [DllImport("LibTorchSharp")]
        internal static extern float THSBFloat16_op_add_lfloat(float a, BFloat16 b);
        [DllImport("LibTorchSharp")]
        internal static extern float THSBFloat16_op_sub_lfloat(float a, BFloat16 b);
        [DllImport("LibTorchSharp")]
        internal static extern float THSBFloat16_op_mul_lfloat(float a, BFloat16 b);
        [DllImport("LibTorchSharp")]
        internal static extern float THSBFloat16_op_div_lfloat(float a, BFloat16 b);
        [DllImport("LibTorchSharp")]
        internal static extern double THSBFloat16_op_add_double(BFloat16 a, double b);
        [DllImport("LibTorchSharp")]
        internal static extern double THSBFloat16_op_sub_double(BFloat16 a, double b);
        [DllImport("LibTorchSharp")]
        internal static extern double THSBFloat16_op_mul_double(BFloat16 a, double b);
        [DllImport("LibTorchSharp")]
        internal static extern double THSBFloat16_op_div_double(BFloat16 a, double b);
        [DllImport("LibTorchSharp")]
        internal static extern double THSBFloat16_op_add_ldouble(double a, BFloat16 b);
        [DllImport("LibTorchSharp")]
        internal static extern double THSBFloat16_op_sub_ldouble(double a, BFloat16 b);
        [DllImport("LibTorchSharp")]
        internal static extern double THSBFloat16_op_mul_ldouble(double a, BFloat16 b);
        [DllImport("LibTorchSharp")]
        internal static extern double THSBFloat16_op_div_ldouble(double a, BFloat16 b);
        [DllImport("LibTorchSharp")]
        internal static extern BFloat16 THSBFloat16_min(BFloat16 bf16);
        [DllImport("LibTorchSharp")]
        internal static extern BFloat16 THSBFloat16_lowest(BFloat16 bf16);
        [DllImport("LibTorchSharp")]
        internal static extern BFloat16 THSBFloat16_max(BFloat16 bf16);
        [DllImport("LibTorchSharp")]
        internal static extern BFloat16 THSBFloat16_epsilon(BFloat16 bf16);
        [DllImport("LibTorchSharp")]
        internal static extern BFloat16 THSBFloat16_round_error(BFloat16 bf16);
        [DllImport("LibTorchSharp")]
        internal static extern BFloat16 THSBFloat16_infinity(BFloat16 bf16);
        [DllImport("LibTorchSharp")]
        internal static extern BFloat16 THSBFloat16_quiet_NaN(BFloat16 bf16);
        [DllImport("LibTorchSharp")]
        internal static extern BFloat16 THSBFloat16_signaling_NaN(BFloat16 bf16);
        [DllImport("LibTorchSharp")]
        internal static extern BFloat16 THSBFloat16_denorm_min(BFloat16 bf16);
    }
}
