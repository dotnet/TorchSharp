
using System;
using System.Runtime.InteropServices;

namespace AtenSharp.Raw {

    // High-performance vector anad matrix operations.
    internal static class Blas {

        // *** Level 1

        // Swap x and y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(swap)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THByteBlas_swap(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate x = a * x
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(scal)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx);
        [DllImport ("caffe2")]
        internal extern static void THByteBlas_scal(
            long n,
            byte a,
            IntPtr /* scalar_t* */ x,
            long incx);

        // Copy x into y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(copy)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THByteBlas_copy(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate y = a * x + y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(axpy)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THByteBlas_axpy(
            long n,
            byte a,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate the dot product.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API scalar_t THBlas_(dot)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static byte THByteBlas_dot(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // *** Level 2

        // Matrix-vector multiplication.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(gemv)(
        //     char trans, int64_t m, int64_t n, scalar_t alpha,
        //     scalar_t *a, int64_t lda, scalar_t *x, int64_t incx,
        //     scalar_t beta, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THByteBlas_gemv(
            byte trans,
            long m,
            long n,
            byte alpha,
            IntPtr /* scalar_t* */ a,
            long lda,
            IntPtr /* scalar_t* */ x,
            long incx,
            byte beta,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Rank 1 operation A := alpha * x * y' + A
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(ger)(
        //     int64_t m, int64_t n, scalar_t alpha, scalar_t *x, int64_t incx,
        //     scalar_t *y, int64_t incy, scalar_t *a, int64_t lda);
        [DllImport ("caffe2")]
        internal extern static void THByteBlas_ger(
            long m,
            long n,
            byte alpha,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy,
            IntPtr /* scalar_t* */ a,
            long lda);

        // *** Level 3

        // Matrix-matrix multiplication.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(gemm)(
        //     char transa, char transb, int64_t m, int64_t n, int64_t k,
        //     scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb,
        //     scalar_t beta, scalar_t *c, int64_t ldc);
        [DllImport ("caffe2")]
        internal extern static void THByteBlas_gemm(
            byte transa,
            byte transb,
            long m,
            long n,
            long k,
            byte alpha,
            IntPtr /* scalar_t* */ a,
            long lda,
            IntPtr /* scalar_t* */ b,
            long ldb,
            byte beta,
            IntPtr /* scalar_t* */ c,
            long ldc);

        // *** Level 1

        // Swap x and y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(swap)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THShortBlas_swap(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate x = a * x
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(scal)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx);
        [DllImport ("caffe2")]
        internal extern static void THShortBlas_scal(
            long n,
            short a,
            IntPtr /* scalar_t* */ x,
            long incx);

        // Copy x into y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(copy)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THShortBlas_copy(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate y = a * x + y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(axpy)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THShortBlas_axpy(
            long n,
            short a,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate the dot product.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API scalar_t THBlas_(dot)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static short THShortBlas_dot(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // *** Level 2

        // Matrix-vector multiplication.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(gemv)(
        //     char trans, int64_t m, int64_t n, scalar_t alpha,
        //     scalar_t *a, int64_t lda, scalar_t *x, int64_t incx,
        //     scalar_t beta, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THShortBlas_gemv(
            byte trans,
            long m,
            long n,
            short alpha,
            IntPtr /* scalar_t* */ a,
            long lda,
            IntPtr /* scalar_t* */ x,
            long incx,
            short beta,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Rank 1 operation A := alpha * x * y' + A
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(ger)(
        //     int64_t m, int64_t n, scalar_t alpha, scalar_t *x, int64_t incx,
        //     scalar_t *y, int64_t incy, scalar_t *a, int64_t lda);
        [DllImport ("caffe2")]
        internal extern static void THShortBlas_ger(
            long m,
            long n,
            short alpha,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy,
            IntPtr /* scalar_t* */ a,
            long lda);

        // *** Level 3

        // Matrix-matrix multiplication.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(gemm)(
        //     char transa, char transb, int64_t m, int64_t n, int64_t k,
        //     scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb,
        //     scalar_t beta, scalar_t *c, int64_t ldc);
        [DllImport ("caffe2")]
        internal extern static void THShortBlas_gemm(
            byte transa,
            byte transb,
            long m,
            long n,
            long k,
            short alpha,
            IntPtr /* scalar_t* */ a,
            long lda,
            IntPtr /* scalar_t* */ b,
            long ldb,
            short beta,
            IntPtr /* scalar_t* */ c,
            long ldc);

        // *** Level 1

        // Swap x and y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(swap)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THIntBlas_swap(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate x = a * x
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(scal)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx);
        [DllImport ("caffe2")]
        internal extern static void THIntBlas_scal(
            long n,
            int a,
            IntPtr /* scalar_t* */ x,
            long incx);

        // Copy x into y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(copy)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THIntBlas_copy(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate y = a * x + y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(axpy)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THIntBlas_axpy(
            long n,
            int a,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate the dot product.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API scalar_t THBlas_(dot)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static int THIntBlas_dot(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // *** Level 2

        // Matrix-vector multiplication.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(gemv)(
        //     char trans, int64_t m, int64_t n, scalar_t alpha,
        //     scalar_t *a, int64_t lda, scalar_t *x, int64_t incx,
        //     scalar_t beta, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THIntBlas_gemv(
            byte trans,
            long m,
            long n,
            int alpha,
            IntPtr /* scalar_t* */ a,
            long lda,
            IntPtr /* scalar_t* */ x,
            long incx,
            int beta,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Rank 1 operation A := alpha * x * y' + A
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(ger)(
        //     int64_t m, int64_t n, scalar_t alpha, scalar_t *x, int64_t incx,
        //     scalar_t *y, int64_t incy, scalar_t *a, int64_t lda);
        [DllImport ("caffe2")]
        internal extern static void THIntBlas_ger(
            long m,
            long n,
            int alpha,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy,
            IntPtr /* scalar_t* */ a,
            long lda);

        // *** Level 3

        // Matrix-matrix multiplication.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(gemm)(
        //     char transa, char transb, int64_t m, int64_t n, int64_t k,
        //     scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb,
        //     scalar_t beta, scalar_t *c, int64_t ldc);
        [DllImport ("caffe2")]
        internal extern static void THIntBlas_gemm(
            byte transa,
            byte transb,
            long m,
            long n,
            long k,
            int alpha,
            IntPtr /* scalar_t* */ a,
            long lda,
            IntPtr /* scalar_t* */ b,
            long ldb,
            int beta,
            IntPtr /* scalar_t* */ c,
            long ldc);

        // *** Level 1

        // Swap x and y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(swap)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THLongBlas_swap(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate x = a * x
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(scal)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx);
        [DllImport ("caffe2")]
        internal extern static void THLongBlas_scal(
            long n,
            long a,
            IntPtr /* scalar_t* */ x,
            long incx);

        // Copy x into y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(copy)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THLongBlas_copy(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate y = a * x + y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(axpy)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THLongBlas_axpy(
            long n,
            long a,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate the dot product.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API scalar_t THBlas_(dot)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static long THLongBlas_dot(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // *** Level 2

        // Matrix-vector multiplication.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(gemv)(
        //     char trans, int64_t m, int64_t n, scalar_t alpha,
        //     scalar_t *a, int64_t lda, scalar_t *x, int64_t incx,
        //     scalar_t beta, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THLongBlas_gemv(
            byte trans,
            long m,
            long n,
            long alpha,
            IntPtr /* scalar_t* */ a,
            long lda,
            IntPtr /* scalar_t* */ x,
            long incx,
            long beta,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Rank 1 operation A := alpha * x * y' + A
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(ger)(
        //     int64_t m, int64_t n, scalar_t alpha, scalar_t *x, int64_t incx,
        //     scalar_t *y, int64_t incy, scalar_t *a, int64_t lda);
        [DllImport ("caffe2")]
        internal extern static void THLongBlas_ger(
            long m,
            long n,
            long alpha,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy,
            IntPtr /* scalar_t* */ a,
            long lda);

        // *** Level 3

        // Matrix-matrix multiplication.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(gemm)(
        //     char transa, char transb, int64_t m, int64_t n, int64_t k,
        //     scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb,
        //     scalar_t beta, scalar_t *c, int64_t ldc);
        [DllImport ("caffe2")]
        internal extern static void THLongBlas_gemm(
            byte transa,
            byte transb,
            long m,
            long n,
            long k,
            long alpha,
            IntPtr /* scalar_t* */ a,
            long lda,
            IntPtr /* scalar_t* */ b,
            long ldb,
            long beta,
            IntPtr /* scalar_t* */ c,
            long ldc);

        // *** Level 1

        // Swap x and y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(swap)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THDoubleBlas_swap(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate x = a * x
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(scal)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx);
        [DllImport ("caffe2")]
        internal extern static void THDoubleBlas_scal(
            long n,
            double a,
            IntPtr /* scalar_t* */ x,
            long incx);

        // Copy x into y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(copy)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THDoubleBlas_copy(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate y = a * x + y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(axpy)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THDoubleBlas_axpy(
            long n,
            double a,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate the dot product.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API scalar_t THBlas_(dot)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static double THDoubleBlas_dot(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // *** Level 2

        // Matrix-vector multiplication.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(gemv)(
        //     char trans, int64_t m, int64_t n, scalar_t alpha,
        //     scalar_t *a, int64_t lda, scalar_t *x, int64_t incx,
        //     scalar_t beta, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THDoubleBlas_gemv(
            byte trans,
            long m,
            long n,
            double alpha,
            IntPtr /* scalar_t* */ a,
            long lda,
            IntPtr /* scalar_t* */ x,
            long incx,
            double beta,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Rank 1 operation A := alpha * x * y' + A
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(ger)(
        //     int64_t m, int64_t n, scalar_t alpha, scalar_t *x, int64_t incx,
        //     scalar_t *y, int64_t incy, scalar_t *a, int64_t lda);
        [DllImport ("caffe2")]
        internal extern static void THDoubleBlas_ger(
            long m,
            long n,
            double alpha,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy,
            IntPtr /* scalar_t* */ a,
            long lda);

        // *** Level 3

        // Matrix-matrix multiplication.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(gemm)(
        //     char transa, char transb, int64_t m, int64_t n, int64_t k,
        //     scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb,
        //     scalar_t beta, scalar_t *c, int64_t ldc);
        [DllImport ("caffe2")]
        internal extern static void THDoubleBlas_gemm(
            byte transa,
            byte transb,
            long m,
            long n,
            long k,
            double alpha,
            IntPtr /* scalar_t* */ a,
            long lda,
            IntPtr /* scalar_t* */ b,
            long ldb,
            double beta,
            IntPtr /* scalar_t* */ c,
            long ldc);

        // *** Level 1

        // Swap x and y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(swap)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THFloatBlas_swap(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate x = a * x
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(scal)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx);
        [DllImport ("caffe2")]
        internal extern static void THFloatBlas_scal(
            long n,
            float a,
            IntPtr /* scalar_t* */ x,
            long incx);

        // Copy x into y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(copy)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THFloatBlas_copy(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate y = a * x + y.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(axpy)(
        //     int64_t n, scalar_t a, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THFloatBlas_axpy(
            long n,
            float a,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Evaluate the dot product.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API scalar_t THBlas_(dot)(
        //     int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static float THFloatBlas_dot(
            long n,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy);

        // *** Level 2

        // Matrix-vector multiplication.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(gemv)(
        //     char trans, int64_t m, int64_t n, scalar_t alpha,
        //     scalar_t *a, int64_t lda, scalar_t *x, int64_t incx,
        //     scalar_t beta, scalar_t *y, int64_t incy);
        [DllImport ("caffe2")]
        internal extern static void THFloatBlas_gemv(
            byte trans,
            long m,
            long n,
            float alpha,
            IntPtr /* scalar_t* */ a,
            long lda,
            IntPtr /* scalar_t* */ x,
            long incx,
            float beta,
            IntPtr /* scalar_t* */ y,
            long incy);

        // Rank 1 operation A := alpha * x * y' + A
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(ger)(
        //     int64_t m, int64_t n, scalar_t alpha, scalar_t *x, int64_t incx,
        //     scalar_t *y, int64_t incy, scalar_t *a, int64_t lda);
        [DllImport ("caffe2")]
        internal extern static void THFloatBlas_ger(
            long m,
            long n,
            float alpha,
            IntPtr /* scalar_t* */ x,
            long incx,
            IntPtr /* scalar_t* */ y,
            long incy,
            IntPtr /* scalar_t* */ a,
            long lda);

        // *** Level 3

        // Matrix-matrix multiplication.
        //
        // Corresponds to the following TH declaration:
        //
        // TH_API void THBlas_(gemm)(
        //     char transa, char transb, int64_t m, int64_t n, int64_t k,
        //     scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb,
        //     scalar_t beta, scalar_t *c, int64_t ldc);
        [DllImport ("caffe2")]
        internal extern static void THFloatBlas_gemm(
            byte transa,
            byte transb,
            long m,
            long n,
            long k,
            float alpha,
            IntPtr /* scalar_t* */ a,
            long lda,
            IntPtr /* scalar_t* */ b,
            long ldb,
            float beta,
            IntPtr /* scalar_t* */ c,
            long ldc);
    }
}