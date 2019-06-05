
using System;
using System.Runtime.InteropServices;

namespace AtenSharp.Raw
{

    // Element-wise operations for vectors.
    // Each scalar_t* pointer can be an offset, and ptrdiff_t
    // parameters usually has number of elements to operate on.
    internal static class Vector
    {

        // Assign value c to n elements of the vector, starting from pointer x.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(fill)(scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THByteVector_fill(
            IntPtr /* scalar_t* */ x, byte c, int /* ptrdiff_t */ n);

        // z = (x + y) + c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cadd)(scalar_t *z, const scalar_t *x, const scalar_t *y, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THByteVector_cadd(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, byte c, int /* ptrdiff_t */ n);

        // z = x * y element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cmul)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THByteVector_cmul(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, int /* ptrdiff_t */ n);

        // z = x / y element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cdiv)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THByteVector_cdiv(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, int /* ptrdiff_t */ n);

        // y = x + c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(adds)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THByteVector_adds(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, byte c, int /* ptrdiff_t */ n);

        // y = x * c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(muls)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THByteVector_muls(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, byte c, int /* ptrdiff_t */ n);

        // z = (x / y) * c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(divs)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THByteVector_divs(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, byte c, int /* ptrdiff_t */ n);

        // copy y = x element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(copy)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THByteVector_copy(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // negate and copy y = -x element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(neg)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THByteVector_neg(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // Fill vector with random values drawn from a normal distribution.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(normal_fill)(
        //     scalar_t *data, const int64_t size, struct THGenerator *generator,
        //     const scalar_t mean, const scalar_t stddev);
        [DllImport("caffe2")]
        internal extern static void THByteVector_normal_fill(
            IntPtr /* scalar_t* */ data, long size, IntPtr /* struct THGenerator* */ generator,
            byte mean, byte stddev);

        // Assign value c to n elements of the vector, starting from pointer x.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(fill)(scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THShortVector_fill(
            IntPtr /* scalar_t* */ x, short c, int /* ptrdiff_t */ n);

        // z = (x + y) + c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cadd)(scalar_t *z, const scalar_t *x, const scalar_t *y, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THShortVector_cadd(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, short c, int /* ptrdiff_t */ n);

        // z = x * y element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cmul)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THShortVector_cmul(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, int /* ptrdiff_t */ n);

        // z = x / y element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cdiv)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THShortVector_cdiv(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, int /* ptrdiff_t */ n);

        // y = x + c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(adds)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THShortVector_adds(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, short c, int /* ptrdiff_t */ n);

        // y = x * c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(muls)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THShortVector_muls(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, short c, int /* ptrdiff_t */ n);

        // z = (x / y) * c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(divs)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THShortVector_divs(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, short c, int /* ptrdiff_t */ n);

        // copy y = x element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(copy)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THShortVector_copy(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // negate and copy y = -x element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(neg)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THShortVector_neg(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // Fill vector with random values drawn from a normal distribution.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(normal_fill)(
        //     scalar_t *data, const int64_t size, struct THGenerator *generator,
        //     const scalar_t mean, const scalar_t stddev);
        [DllImport("caffe2")]
        internal extern static void THShortVector_normal_fill(
            IntPtr /* scalar_t* */ data, long size, IntPtr /* struct THGenerator* */ generator,
            short mean, short stddev);

        // y = |x| element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(abs)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THShortVector_abs(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // Assign value c to n elements of the vector, starting from pointer x.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(fill)(scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THIntVector_fill(
            IntPtr /* scalar_t* */ x, int c, int /* ptrdiff_t */ n);

        // z = (x + y) + c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cadd)(scalar_t *z, const scalar_t *x, const scalar_t *y, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THIntVector_cadd(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, int c, int /* ptrdiff_t */ n);

        // z = x * y element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cmul)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THIntVector_cmul(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, int /* ptrdiff_t */ n);

        // z = x / y element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cdiv)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THIntVector_cdiv(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, int /* ptrdiff_t */ n);

        // y = x + c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(adds)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THIntVector_adds(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int c, int /* ptrdiff_t */ n);

        // y = x * c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(muls)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THIntVector_muls(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int c, int /* ptrdiff_t */ n);

        // z = (x / y) * c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(divs)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THIntVector_divs(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int c, int /* ptrdiff_t */ n);

        // copy y = x element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(copy)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THIntVector_copy(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // negate and copy y = -x element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(neg)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THIntVector_neg(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // Fill vector with random values drawn from a normal distribution.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(normal_fill)(
        //     scalar_t *data, const int64_t size, struct THGenerator *generator,
        //     const scalar_t mean, const scalar_t stddev);
        [DllImport("caffe2")]
        internal extern static void THIntVector_normal_fill(
            IntPtr /* scalar_t* */ data, long size, IntPtr /* struct THGenerator* */ generator,
            int mean, int stddev);

        // y = |x| element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(abs)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THIntVector_abs(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // Assign value c to n elements of the vector, starting from pointer x.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(fill)(scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THLongVector_fill(
            IntPtr /* scalar_t* */ x, long c, int /* ptrdiff_t */ n);

        // z = (x + y) + c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cadd)(scalar_t *z, const scalar_t *x, const scalar_t *y, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THLongVector_cadd(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, long c, int /* ptrdiff_t */ n);

        // z = x * y element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cmul)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THLongVector_cmul(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, int /* ptrdiff_t */ n);

        // z = x / y element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cdiv)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THLongVector_cdiv(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, int /* ptrdiff_t */ n);

        // y = x + c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(adds)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THLongVector_adds(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, long c, int /* ptrdiff_t */ n);

        // y = x * c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(muls)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THLongVector_muls(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, long c, int /* ptrdiff_t */ n);

        // z = (x / y) * c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(divs)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THLongVector_divs(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, long c, int /* ptrdiff_t */ n);

        // copy y = x element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(copy)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THLongVector_copy(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // negate and copy y = -x element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(neg)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THLongVector_neg(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // Fill vector with random values drawn from a normal distribution.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(normal_fill)(
        //     scalar_t *data, const int64_t size, struct THGenerator *generator,
        //     const scalar_t mean, const scalar_t stddev);
        [DllImport("caffe2")]
        internal extern static void THLongVector_normal_fill(
            IntPtr /* scalar_t* */ data, long size, IntPtr /* struct THGenerator* */ generator,
            long mean, long stddev);

        // y = |x| element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(abs)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THLongVector_abs(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // Assign value c to n elements of the vector, starting from pointer x.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(fill)(scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_fill(
            IntPtr /* scalar_t* */ x, double c, int /* ptrdiff_t */ n);

        // z = (x + y) + c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cadd)(scalar_t *z, const scalar_t *x, const scalar_t *y, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_cadd(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, double c, int /* ptrdiff_t */ n);

        // z = x * y element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cmul)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_cmul(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, int /* ptrdiff_t */ n);

        // z = x / y element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cdiv)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_cdiv(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, int /* ptrdiff_t */ n);

        // y = x + c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(adds)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_adds(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, double c, int /* ptrdiff_t */ n);

        // y = x * c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(muls)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_muls(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, double c, int /* ptrdiff_t */ n);

        // z = (x / y) * c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(divs)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_divs(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, double c, int /* ptrdiff_t */ n);

        // copy y = x element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(copy)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_copy(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // negate and copy y = -x element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(neg)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_neg(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // Fill vector with random values drawn from a normal distribution.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(normal_fill)(
        //     scalar_t *data, const int64_t size, struct THGenerator *generator,
        //     const scalar_t mean, const scalar_t stddev);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_normal_fill(
            IntPtr /* scalar_t* */ data, long size, IntPtr /* struct THGenerator* */ generator,
            double mean, double stddev);

        // y = x^c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(pow)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_pow(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, double c, int /* ptrdiff_t */ n);

        // y = log(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(log)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_log(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = lgamma(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(lgamma)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_lgamma(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = digamma(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(digamma)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_digamma(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = trigamma(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(trigamma)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_trigamma(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = log10(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(log10)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_log10(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = log1p(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(log1p)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_log1p(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = log2(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(log2)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_log2(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = sigmoid(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(sigmoid)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_sigmoid(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = exp(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(exp)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_exp(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = expm1(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(expm1)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_expm1(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = erf(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(erf)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_erf(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = erfc(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(erfc)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_erfc(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = erfinv(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(erfinv)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_erfinv(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = cos(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cos)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_cos(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = acos(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(acos)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_acos(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = cosh(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cosh)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_cosh(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = sin(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(sin)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_sin(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = asin(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(asin)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_asin(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = sinh(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(sinh)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_sinh(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = tan(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(tan)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_tan(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = atan(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(atan)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_atan(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = tanh(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(tanh)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_tanh(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = sqrt(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(sqrt)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_sqrt(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = rsqrt(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(rsqrt)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_rsqrt(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = ceil(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(ceil)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_ceil(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = floor(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(floor)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_floor(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = round(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(round)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_round(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = abs(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(abs)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_abs(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = trunc(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(trunc)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_trunc(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = frac(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(frac)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_frac(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = cinv(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cinv)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THDoubleVector_cinv(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);
        // Assign value c to n elements of the vector, starting from pointer x.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(fill)(scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_fill(
            IntPtr /* scalar_t* */ x, float c, int /* ptrdiff_t */ n);

        // z = (x + y) + c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cadd)(scalar_t *z, const scalar_t *x, const scalar_t *y, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_cadd(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, float c, int /* ptrdiff_t */ n);

        // z = x * y element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cmul)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_cmul(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, int /* ptrdiff_t */ n);

        // z = x / y element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cdiv)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_cdiv(
            IntPtr /* scalar_t* */ z, IntPtr /* scalar_t* */ x, IntPtr /* scalar_t* */ y, int /* ptrdiff_t */ n);

        // y = x + c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(adds)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_adds(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, float c, int /* ptrdiff_t */ n);

        // y = x * c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(muls)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_muls(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, float c, int /* ptrdiff_t */ n);

        // z = (x / y) * c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(divs)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_divs(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, float c, int /* ptrdiff_t */ n);

        // copy y = x element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(copy)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_copy(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // negate and copy y = -x element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(neg)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_neg(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // Fill vector with random values drawn from a normal distribution.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(normal_fill)(
        //     scalar_t *data, const int64_t size, struct THGenerator *generator,
        //     const scalar_t mean, const scalar_t stddev);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_normal_fill(
            IntPtr /* scalar_t* */ data, long size, IntPtr /* struct THGenerator* */ generator,
            float mean, float stddev);

        // y = x^c element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(pow)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_pow(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, float c, int /* ptrdiff_t */ n);

        // y = log(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(log)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_log(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = lgamma(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(lgamma)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_lgamma(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = digamma(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(digamma)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_digamma(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = trigamma(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(trigamma)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_trigamma(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = log10(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(log10)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_log10(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = log1p(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(log1p)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_log1p(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = log2(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(log2)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_log2(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = sigmoid(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(sigmoid)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_sigmoid(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = exp(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(exp)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_exp(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = expm1(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(expm1)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_expm1(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = erf(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(erf)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_erf(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = erfc(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(erfc)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_erfc(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = erfinv(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(erfinv)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_erfinv(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = cos(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cos)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_cos(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = acos(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(acos)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_acos(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = cosh(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cosh)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_cosh(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = sin(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(sin)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_sin(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = asin(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(asin)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_asin(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = sinh(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(sinh)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_sinh(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = tan(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(tan)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_tan(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = atan(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(atan)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_atan(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = tanh(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(tanh)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_tanh(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = sqrt(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(sqrt)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_sqrt(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = rsqrt(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(rsqrt)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_rsqrt(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = ceil(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(ceil)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_ceil(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = floor(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(floor)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_floor(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = round(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(round)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_round(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = abs(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(abs)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_abs(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = trunc(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(trunc)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_trunc(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = frac(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(frac)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_frac(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);

        // y = cinv(x) element-wise for n elements
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THVector_(cinv)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
        [DllImport("caffe2")]
        internal extern static void THFloatVector_cinv(
            IntPtr /* scalar_t* */ y, IntPtr /* scalar_t* */ x, int /* ptrdiff_t */ n);
    }
}
