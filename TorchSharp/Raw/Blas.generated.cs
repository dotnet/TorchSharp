
using System;
using System.Runtime.InteropServices;

namespace TorchSharp {
    public partial class ByteTensor : IDisposable {

        // *** Level 1

        // TH_API void THByteBlas_(swap)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THByteBlas_swap(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API void THByteBlas_(scal)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx);
        [DllImport ("caffe2")]
        extern static void THByteBlas_scal(
            long n,
            byte a,
            IntPtr x,
            long incx);

        // TH_API void THByteBlas_(copy)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THByteBlas_copy(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API void THByteBlas_(axpy)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THByteBlas_axpy(
            long n,
            byte a,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API scalar_t THByteBlas_(dot)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static byte THByteBlas_dot(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // *** Level 2

        // TH_API void THByteBlas_(gemv)(
        //     char trans, int64_t m, int64_t n, scalar_t alpha,
        //     scalar_t *a, int64_t lda, scalar_t *x, int64_t incx,
        //     scalar_t beta, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THByteBlas_gemv(
            byte trans,
            long m,
            long n,
            byte alpha,
            IntPtr a,
            long lda,
            IntPtr x,
            long incx,
            byte beta,
            IntPtr y,
            long incy);

        // TH_API void THByteBlas_(ger)(
        //     int64_t m, int64_t n, scalar_t alpha, scalar_t *x, int64_t incx,
        //     scalar_t *y, int64_t incy, scalar_t *a, int64_t lda);
        [DllImport ("caffe2")]
        extern static void THByteBlas_ger(
            long m,
            long n,
            byte alpha,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy,
            IntPtr a,
            long lda);

        // *** Level 3

        // TH_API void THByteBlas_(gemm)(
        //     char transa, char transb, int64_t m, int64_t n, int64_t k,
        //     scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb,
        //     scalar_t beta, scalar_t *c, int64_t ldc);
        [DllImport ("caffe2")]
        extern static void THByteBlas_gemm(
            byte transa,
            byte transb,
            long m,
            long n,
            long k,
            byte alpha,
            IntPtr a,
            long lda,
            IntPtr b,
            long ldb,
            byte beta,
            IntPtr c,
            long ldc);
    } // class ByteTensor
    public partial class ShortTensor : IDisposable {

        // *** Level 1

        // TH_API void THShortBlas_(swap)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THShortBlas_swap(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API void THShortBlas_(scal)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx);
        [DllImport ("caffe2")]
        extern static void THShortBlas_scal(
            long n,
            short a,
            IntPtr x,
            long incx);

        // TH_API void THShortBlas_(copy)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THShortBlas_copy(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API void THShortBlas_(axpy)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THShortBlas_axpy(
            long n,
            short a,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API scalar_t THShortBlas_(dot)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static short THShortBlas_dot(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // *** Level 2

        // TH_API void THShortBlas_(gemv)(
        //     char trans, int64_t m, int64_t n, scalar_t alpha,
        //     scalar_t *a, int64_t lda, scalar_t *x, int64_t incx,
        //     scalar_t beta, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THShortBlas_gemv(
            byte trans,
            long m,
            long n,
            short alpha,
            IntPtr a,
            long lda,
            IntPtr x,
            long incx,
            short beta,
            IntPtr y,
            long incy);

        // TH_API void THShortBlas_(ger)(
        //     int64_t m, int64_t n, scalar_t alpha, scalar_t *x, int64_t incx,
        //     scalar_t *y, int64_t incy, scalar_t *a, int64_t lda);
        [DllImport ("caffe2")]
        extern static void THShortBlas_ger(
            long m,
            long n,
            short alpha,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy,
            IntPtr a,
            long lda);

        // *** Level 3

        // TH_API void THShortBlas_(gemm)(
        //     char transa, char transb, int64_t m, int64_t n, int64_t k,
        //     scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb,
        //     scalar_t beta, scalar_t *c, int64_t ldc);
        [DllImport ("caffe2")]
        extern static void THShortBlas_gemm(
            byte transa,
            byte transb,
            long m,
            long n,
            long k,
            short alpha,
            IntPtr a,
            long lda,
            IntPtr b,
            long ldb,
            short beta,
            IntPtr c,
            long ldc);
    } // class ShortTensor
    public partial class IntTensor : IDisposable {

        // *** Level 1

        // TH_API void THIntBlas_(swap)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THIntBlas_swap(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API void THIntBlas_(scal)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx);
        [DllImport ("caffe2")]
        extern static void THIntBlas_scal(
            long n,
            int a,
            IntPtr x,
            long incx);

        // TH_API void THIntBlas_(copy)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THIntBlas_copy(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API void THIntBlas_(axpy)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THIntBlas_axpy(
            long n,
            int a,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API scalar_t THIntBlas_(dot)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static int THIntBlas_dot(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // *** Level 2

        // TH_API void THIntBlas_(gemv)(
        //     char trans, int64_t m, int64_t n, scalar_t alpha,
        //     scalar_t *a, int64_t lda, scalar_t *x, int64_t incx,
        //     scalar_t beta, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THIntBlas_gemv(
            byte trans,
            long m,
            long n,
            int alpha,
            IntPtr a,
            long lda,
            IntPtr x,
            long incx,
            int beta,
            IntPtr y,
            long incy);

        // TH_API void THIntBlas_(ger)(
        //     int64_t m, int64_t n, scalar_t alpha, scalar_t *x, int64_t incx,
        //     scalar_t *y, int64_t incy, scalar_t *a, int64_t lda);
        [DllImport ("caffe2")]
        extern static void THIntBlas_ger(
            long m,
            long n,
            int alpha,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy,
            IntPtr a,
            long lda);

        // *** Level 3

        // TH_API void THIntBlas_(gemm)(
        //     char transa, char transb, int64_t m, int64_t n, int64_t k,
        //     scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb,
        //     scalar_t beta, scalar_t *c, int64_t ldc);
        [DllImport ("caffe2")]
        extern static void THIntBlas_gemm(
            byte transa,
            byte transb,
            long m,
            long n,
            long k,
            int alpha,
            IntPtr a,
            long lda,
            IntPtr b,
            long ldb,
            int beta,
            IntPtr c,
            long ldc);
    } // class IntTensor
    public partial class LongTensor : IDisposable {

        // *** Level 1

        // TH_API void THLongBlas_(swap)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THLongBlas_swap(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API void THLongBlas_(scal)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx);
        [DllImport ("caffe2")]
        extern static void THLongBlas_scal(
            long n,
            long a,
            IntPtr x,
            long incx);

        // TH_API void THLongBlas_(copy)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THLongBlas_copy(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API void THLongBlas_(axpy)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THLongBlas_axpy(
            long n,
            long a,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API scalar_t THLongBlas_(dot)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static long THLongBlas_dot(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // *** Level 2

        // TH_API void THLongBlas_(gemv)(
        //     char trans, int64_t m, int64_t n, scalar_t alpha,
        //     scalar_t *a, int64_t lda, scalar_t *x, int64_t incx,
        //     scalar_t beta, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THLongBlas_gemv(
            byte trans,
            long m,
            long n,
            long alpha,
            IntPtr a,
            long lda,
            IntPtr x,
            long incx,
            long beta,
            IntPtr y,
            long incy);

        // TH_API void THLongBlas_(ger)(
        //     int64_t m, int64_t n, scalar_t alpha, scalar_t *x, int64_t incx,
        //     scalar_t *y, int64_t incy, scalar_t *a, int64_t lda);
        [DllImport ("caffe2")]
        extern static void THLongBlas_ger(
            long m,
            long n,
            long alpha,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy,
            IntPtr a,
            long lda);

        // *** Level 3

        // TH_API void THLongBlas_(gemm)(
        //     char transa, char transb, int64_t m, int64_t n, int64_t k,
        //     scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb,
        //     scalar_t beta, scalar_t *c, int64_t ldc);
        [DllImport ("caffe2")]
        extern static void THLongBlas_gemm(
            byte transa,
            byte transb,
            long m,
            long n,
            long k,
            long alpha,
            IntPtr a,
            long lda,
            IntPtr b,
            long ldb,
            long beta,
            IntPtr c,
            long ldc);
    } // class LongTensor
    public partial class DoubleTensor : IDisposable {

        // *** Level 1

        // TH_API void THDoubleBlas_(swap)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THDoubleBlas_swap(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API void THDoubleBlas_(scal)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx);
        [DllImport ("caffe2")]
        extern static void THDoubleBlas_scal(
            long n,
            double a,
            IntPtr x,
            long incx);

        // TH_API void THDoubleBlas_(copy)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THDoubleBlas_copy(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API void THDoubleBlas_(axpy)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THDoubleBlas_axpy(
            long n,
            double a,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API scalar_t THDoubleBlas_(dot)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static double THDoubleBlas_dot(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // *** Level 2

        // TH_API void THDoubleBlas_(gemv)(
        //     char trans, int64_t m, int64_t n, scalar_t alpha,
        //     scalar_t *a, int64_t lda, scalar_t *x, int64_t incx,
        //     scalar_t beta, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THDoubleBlas_gemv(
            byte trans,
            long m,
            long n,
            double alpha,
            IntPtr a,
            long lda,
            IntPtr x,
            long incx,
            double beta,
            IntPtr y,
            long incy);

        // TH_API void THDoubleBlas_(ger)(
        //     int64_t m, int64_t n, scalar_t alpha, scalar_t *x, int64_t incx,
        //     scalar_t *y, int64_t incy, scalar_t *a, int64_t lda);
        [DllImport ("caffe2")]
        extern static void THDoubleBlas_ger(
            long m,
            long n,
            double alpha,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy,
            IntPtr a,
            long lda);

        // *** Level 3

        // TH_API void THDoubleBlas_(gemm)(
        //     char transa, char transb, int64_t m, int64_t n, int64_t k,
        //     scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb,
        //     scalar_t beta, scalar_t *c, int64_t ldc);
        [DllImport ("caffe2")]
        extern static void THDoubleBlas_gemm(
            byte transa,
            byte transb,
            long m,
            long n,
            long k,
            double alpha,
            IntPtr a,
            long lda,
            IntPtr b,
            long ldb,
            double beta,
            IntPtr c,
            long ldc);
    } // class DoubleTensor
    public partial class FloatTensor : IDisposable {

        // *** Level 1

        // TH_API void THFloatBlas_(swap)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THFloatBlas_swap(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API void THFloatBlas_(scal)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx);
        [DllImport ("caffe2")]
        extern static void THFloatBlas_scal(
            long n,
            float a,
            IntPtr x,
            long incx);

        // TH_API void THFloatBlas_(copy)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THFloatBlas_copy(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API void THFloatBlas_(axpy)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THFloatBlas_axpy(
            long n,
            float a,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // TH_API scalar_t THFloatBlas_(dot)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static float THFloatBlas_dot(
            long n,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy);

        // *** Level 2

        // TH_API void THFloatBlas_(gemv)(
        //     char trans, int64_t m, int64_t n, scalar_t alpha,
        //     scalar_t *a, int64_t lda, scalar_t *x, int64_t incx,
        //     scalar_t beta, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        extern static void THFloatBlas_gemv(
            byte trans,
            long m,
            long n,
            float alpha,
            IntPtr a,
            long lda,
            IntPtr x,
            long incx,
            float beta,
            IntPtr y,
            long incy);

        // TH_API void THFloatBlas_(ger)(
        //     int64_t m, int64_t n, scalar_t alpha, scalar_t *x, int64_t incx,
        //     scalar_t *y, int64_t incy, scalar_t *a, int64_t lda);
        [DllImport ("caffe2")]
        extern static void THFloatBlas_ger(
            long m,
            long n,
            float alpha,
            IntPtr x,
            long incx,
            IntPtr y,
            long incy,
            IntPtr a,
            long lda);

        // *** Level 3

        // TH_API void THFloatBlas_(gemm)(
        //     char transa, char transb, int64_t m, int64_t n, int64_t k,
        //     scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb,
        //     scalar_t beta, scalar_t *c, int64_t ldc);
        [DllImport ("caffe2")]
        extern static void THFloatBlas_gemm(
            byte transa,
            byte transb,
            long m,
            long n,
            long k,
            float alpha,
            IntPtr a,
            long lda,
            IntPtr b,
            long ldb,
            float beta,
            IntPtr c,
            long ldc);
    } // class FloatTensor
} // namespace TorchSharp
