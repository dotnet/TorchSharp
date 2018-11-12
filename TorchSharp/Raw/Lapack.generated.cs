
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.Raw {

    // High-performance linear algebra operations.
    internal static class Lapack {

        // Solve AX=B
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gesv)(
        //     int n, int nrhs, scalar_t *a, int lda, int *ipiv,
        //      scalar_t *b, int ldb, int* info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_gesv(
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Solve a triangular system of the form A * X = B or A^T * X = B
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(trtrs)(
        //     char uplo, char trans, char diag, int n, int nrhs,
        //     scalar_t *a, int lda, scalar_t *b, int ldb, int* info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_trtrs(
            byte uplo,
            byte trans,
            byte diag,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Evaluate ||AX-B||
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gels)(
        //     char trans, int m, int n, int nrhs, scalar_t *a, int lda,
        //     scalar_t *b, int ldb, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_gels(
            byte trans,
            int m,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Eigenvals
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(syev)(
        //     char jobz, char uplo, int n, scalar_t *a, int lda,
        //     scalar_t *w, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_syev(
            byte jobz,
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ w,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Non-sym eigenvals
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(geev)(
        //     char jobvl, char jobvr, int n, scalar_t *a, int lda,
        //     scalar_t *wr, scalar_t *wi, scalar_t* vl, int ldvl,
        //     scalar_t *vr, int ldvr, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_geev(
            byte jobvl,
            byte jobvr,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ wr,
            IntPtr /* scalar_t* */ wi,
            IntPtr /* scalar_t* */ vl,
            int ldvl,
            IntPtr /* scalar_t* */ vr,
            int ldvr,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // SVD
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gesdd)(
        //     char jobz, int m, int n, scalar_t *a, int lda, scalar_t *s,
        //     scalar_t *u, int ldu, scalar_t *vt, int ldvt,
        //     scalar_t *work, int lwork, int *iwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_gesdd(
            byte jobz,
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ s,
            IntPtr /* scalar_t* */ u,
            int ldu,
            IntPtr /* scalar_t* */ vt,
            int ldvt,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ iwork,
            IntPtr /* int* */ info);

        // LU decomposition
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getrf)(
        //     int m, int n, scalar_t *a, int lda, int *ipiv, int *info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_getrf(
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* int* */ info);

        // LU decomposition
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getrs)(
        //     char trans, int n, int nrhs, scalar_t *a, int lda,
        //     int *ipiv, scalar_t *b, int ldb, int *info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_getrs(
            byte trans,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Matrix Inverse
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getri)(
        //     int n, scalar_t *a, int lda, int *ipiv,
        //     scalar_t *work, int lwork, int* info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_getri(
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // *** Positive Definite matrices

        // Cholesky factorization
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potrf)(
        //     char uplo, int n, scalar_t *a, int lda, int *info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_potrf(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ info);

        // Matrix inverse based on Cholesky factorization
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potri)(
        //     char uplo, int n, scalar_t *a, int lda, int *info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_potri(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ info);

        // Solve A*X = B with a symmetric positive definite matrix A
        // using the Cholesky factorization.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potrs)(
        //     char uplo, int n, int nrhs, scalar_t *a, int lda,
        //     scalar_t *b, int ldb, int *info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_potrs(
            char uplo,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Cholesky factorization with complete pivoting.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(pstrf)(
        //     char uplo, int n, scalar_t *a, int lda, int *piv,
        //     int *rank, scalar_t tol, scalar_t *work, int *info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_pstrf(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ piv,
            IntPtr /* int* */ rank,
            byte tol,
            IntPtr /* scalar_t* */ work,
            IntPtr /* int* */ info);

        // QR decomposition.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(geqrf)(
        //     int m, int n, scalar_t *a, int lda,
        //     scalar_t *tau, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_geqrf(
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Build Q from output of geqrf
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(orgqr)(
        //     int m, int n, int k, scalar_t *a, int lda,
        //     scalar_t *tau, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_orgqr(
            int m,
            int n,
            int k,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Multiply Q with a matrix from output of geqrf
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(ormqr)(
        //     char side, char trans, int m, int n, int k,
        //     scalar_t *a, int lda, scalar_t *tau, scalar_t *c, int ldc,
        //     scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THByteLapack_ormqr(
            byte side,
            byte trans,
            int m,
            int n,
            int k,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ c,
            int ldc,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Solve AX=B
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gesv)(
        //     int n, int nrhs, scalar_t *a, int lda, int *ipiv,
        //      scalar_t *b, int ldb, int* info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_gesv(
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Solve a triangular system of the form A * X = B or A^T * X = B
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(trtrs)(
        //     char uplo, char trans, char diag, int n, int nrhs,
        //     scalar_t *a, int lda, scalar_t *b, int ldb, int* info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_trtrs(
            byte uplo,
            byte trans,
            byte diag,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Evaluate ||AX-B||
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gels)(
        //     char trans, int m, int n, int nrhs, scalar_t *a, int lda,
        //     scalar_t *b, int ldb, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_gels(
            byte trans,
            int m,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Eigenvals
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(syev)(
        //     char jobz, char uplo, int n, scalar_t *a, int lda,
        //     scalar_t *w, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_syev(
            byte jobz,
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ w,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Non-sym eigenvals
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(geev)(
        //     char jobvl, char jobvr, int n, scalar_t *a, int lda,
        //     scalar_t *wr, scalar_t *wi, scalar_t* vl, int ldvl,
        //     scalar_t *vr, int ldvr, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_geev(
            byte jobvl,
            byte jobvr,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ wr,
            IntPtr /* scalar_t* */ wi,
            IntPtr /* scalar_t* */ vl,
            int ldvl,
            IntPtr /* scalar_t* */ vr,
            int ldvr,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // SVD
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gesdd)(
        //     char jobz, int m, int n, scalar_t *a, int lda, scalar_t *s,
        //     scalar_t *u, int ldu, scalar_t *vt, int ldvt,
        //     scalar_t *work, int lwork, int *iwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_gesdd(
            byte jobz,
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ s,
            IntPtr /* scalar_t* */ u,
            int ldu,
            IntPtr /* scalar_t* */ vt,
            int ldvt,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ iwork,
            IntPtr /* int* */ info);

        // LU decomposition
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getrf)(
        //     int m, int n, scalar_t *a, int lda, int *ipiv, int *info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_getrf(
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* int* */ info);

        // LU decomposition
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getrs)(
        //     char trans, int n, int nrhs, scalar_t *a, int lda,
        //     int *ipiv, scalar_t *b, int ldb, int *info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_getrs(
            byte trans,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Matrix Inverse
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getri)(
        //     int n, scalar_t *a, int lda, int *ipiv,
        //     scalar_t *work, int lwork, int* info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_getri(
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // *** Positive Definite matrices

        // Cholesky factorization
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potrf)(
        //     char uplo, int n, scalar_t *a, int lda, int *info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_potrf(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ info);

        // Matrix inverse based on Cholesky factorization
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potri)(
        //     char uplo, int n, scalar_t *a, int lda, int *info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_potri(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ info);

        // Solve A*X = B with a symmetric positive definite matrix A
        // using the Cholesky factorization.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potrs)(
        //     char uplo, int n, int nrhs, scalar_t *a, int lda,
        //     scalar_t *b, int ldb, int *info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_potrs(
            char uplo,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Cholesky factorization with complete pivoting.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(pstrf)(
        //     char uplo, int n, scalar_t *a, int lda, int *piv,
        //     int *rank, scalar_t tol, scalar_t *work, int *info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_pstrf(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ piv,
            IntPtr /* int* */ rank,
            short tol,
            IntPtr /* scalar_t* */ work,
            IntPtr /* int* */ info);

        // QR decomposition.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(geqrf)(
        //     int m, int n, scalar_t *a, int lda,
        //     scalar_t *tau, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_geqrf(
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Build Q from output of geqrf
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(orgqr)(
        //     int m, int n, int k, scalar_t *a, int lda,
        //     scalar_t *tau, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_orgqr(
            int m,
            int n,
            int k,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Multiply Q with a matrix from output of geqrf
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(ormqr)(
        //     char side, char trans, int m, int n, int k,
        //     scalar_t *a, int lda, scalar_t *tau, scalar_t *c, int ldc,
        //     scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THShortLapack_ormqr(
            byte side,
            byte trans,
            int m,
            int n,
            int k,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ c,
            int ldc,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Solve AX=B
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gesv)(
        //     int n, int nrhs, scalar_t *a, int lda, int *ipiv,
        //      scalar_t *b, int ldb, int* info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_gesv(
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Solve a triangular system of the form A * X = B or A^T * X = B
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(trtrs)(
        //     char uplo, char trans, char diag, int n, int nrhs,
        //     scalar_t *a, int lda, scalar_t *b, int ldb, int* info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_trtrs(
            byte uplo,
            byte trans,
            byte diag,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Evaluate ||AX-B||
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gels)(
        //     char trans, int m, int n, int nrhs, scalar_t *a, int lda,
        //     scalar_t *b, int ldb, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_gels(
            byte trans,
            int m,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Eigenvals
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(syev)(
        //     char jobz, char uplo, int n, scalar_t *a, int lda,
        //     scalar_t *w, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_syev(
            byte jobz,
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ w,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Non-sym eigenvals
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(geev)(
        //     char jobvl, char jobvr, int n, scalar_t *a, int lda,
        //     scalar_t *wr, scalar_t *wi, scalar_t* vl, int ldvl,
        //     scalar_t *vr, int ldvr, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_geev(
            byte jobvl,
            byte jobvr,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ wr,
            IntPtr /* scalar_t* */ wi,
            IntPtr /* scalar_t* */ vl,
            int ldvl,
            IntPtr /* scalar_t* */ vr,
            int ldvr,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // SVD
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gesdd)(
        //     char jobz, int m, int n, scalar_t *a, int lda, scalar_t *s,
        //     scalar_t *u, int ldu, scalar_t *vt, int ldvt,
        //     scalar_t *work, int lwork, int *iwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_gesdd(
            byte jobz,
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ s,
            IntPtr /* scalar_t* */ u,
            int ldu,
            IntPtr /* scalar_t* */ vt,
            int ldvt,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ iwork,
            IntPtr /* int* */ info);

        // LU decomposition
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getrf)(
        //     int m, int n, scalar_t *a, int lda, int *ipiv, int *info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_getrf(
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* int* */ info);

        // LU decomposition
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getrs)(
        //     char trans, int n, int nrhs, scalar_t *a, int lda,
        //     int *ipiv, scalar_t *b, int ldb, int *info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_getrs(
            byte trans,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Matrix Inverse
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getri)(
        //     int n, scalar_t *a, int lda, int *ipiv,
        //     scalar_t *work, int lwork, int* info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_getri(
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // *** Positive Definite matrices

        // Cholesky factorization
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potrf)(
        //     char uplo, int n, scalar_t *a, int lda, int *info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_potrf(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ info);

        // Matrix inverse based on Cholesky factorization
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potri)(
        //     char uplo, int n, scalar_t *a, int lda, int *info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_potri(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ info);

        // Solve A*X = B with a symmetric positive definite matrix A
        // using the Cholesky factorization.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potrs)(
        //     char uplo, int n, int nrhs, scalar_t *a, int lda,
        //     scalar_t *b, int ldb, int *info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_potrs(
            char uplo,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Cholesky factorization with complete pivoting.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(pstrf)(
        //     char uplo, int n, scalar_t *a, int lda, int *piv,
        //     int *rank, scalar_t tol, scalar_t *work, int *info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_pstrf(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ piv,
            IntPtr /* int* */ rank,
            int tol,
            IntPtr /* scalar_t* */ work,
            IntPtr /* int* */ info);

        // QR decomposition.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(geqrf)(
        //     int m, int n, scalar_t *a, int lda,
        //     scalar_t *tau, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_geqrf(
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Build Q from output of geqrf
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(orgqr)(
        //     int m, int n, int k, scalar_t *a, int lda,
        //     scalar_t *tau, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_orgqr(
            int m,
            int n,
            int k,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Multiply Q with a matrix from output of geqrf
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(ormqr)(
        //     char side, char trans, int m, int n, int k,
        //     scalar_t *a, int lda, scalar_t *tau, scalar_t *c, int ldc,
        //     scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THIntLapack_ormqr(
            byte side,
            byte trans,
            int m,
            int n,
            int k,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ c,
            int ldc,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Solve AX=B
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gesv)(
        //     int n, int nrhs, scalar_t *a, int lda, int *ipiv,
        //      scalar_t *b, int ldb, int* info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_gesv(
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Solve a triangular system of the form A * X = B or A^T * X = B
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(trtrs)(
        //     char uplo, char trans, char diag, int n, int nrhs,
        //     scalar_t *a, int lda, scalar_t *b, int ldb, int* info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_trtrs(
            byte uplo,
            byte trans,
            byte diag,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Evaluate ||AX-B||
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gels)(
        //     char trans, int m, int n, int nrhs, scalar_t *a, int lda,
        //     scalar_t *b, int ldb, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_gels(
            byte trans,
            int m,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Eigenvals
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(syev)(
        //     char jobz, char uplo, int n, scalar_t *a, int lda,
        //     scalar_t *w, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_syev(
            byte jobz,
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ w,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Non-sym eigenvals
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(geev)(
        //     char jobvl, char jobvr, int n, scalar_t *a, int lda,
        //     scalar_t *wr, scalar_t *wi, scalar_t* vl, int ldvl,
        //     scalar_t *vr, int ldvr, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_geev(
            byte jobvl,
            byte jobvr,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ wr,
            IntPtr /* scalar_t* */ wi,
            IntPtr /* scalar_t* */ vl,
            int ldvl,
            IntPtr /* scalar_t* */ vr,
            int ldvr,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // SVD
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gesdd)(
        //     char jobz, int m, int n, scalar_t *a, int lda, scalar_t *s,
        //     scalar_t *u, int ldu, scalar_t *vt, int ldvt,
        //     scalar_t *work, int lwork, int *iwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_gesdd(
            byte jobz,
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ s,
            IntPtr /* scalar_t* */ u,
            int ldu,
            IntPtr /* scalar_t* */ vt,
            int ldvt,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ iwork,
            IntPtr /* int* */ info);

        // LU decomposition
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getrf)(
        //     int m, int n, scalar_t *a, int lda, int *ipiv, int *info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_getrf(
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* int* */ info);

        // LU decomposition
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getrs)(
        //     char trans, int n, int nrhs, scalar_t *a, int lda,
        //     int *ipiv, scalar_t *b, int ldb, int *info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_getrs(
            byte trans,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Matrix Inverse
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getri)(
        //     int n, scalar_t *a, int lda, int *ipiv,
        //     scalar_t *work, int lwork, int* info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_getri(
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // *** Positive Definite matrices

        // Cholesky factorization
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potrf)(
        //     char uplo, int n, scalar_t *a, int lda, int *info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_potrf(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ info);

        // Matrix inverse based on Cholesky factorization
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potri)(
        //     char uplo, int n, scalar_t *a, int lda, int *info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_potri(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ info);

        // Solve A*X = B with a symmetric positive definite matrix A
        // using the Cholesky factorization.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potrs)(
        //     char uplo, int n, int nrhs, scalar_t *a, int lda,
        //     scalar_t *b, int ldb, int *info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_potrs(
            char uplo,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Cholesky factorization with complete pivoting.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(pstrf)(
        //     char uplo, int n, scalar_t *a, int lda, int *piv,
        //     int *rank, scalar_t tol, scalar_t *work, int *info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_pstrf(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ piv,
            IntPtr /* int* */ rank,
            long tol,
            IntPtr /* scalar_t* */ work,
            IntPtr /* int* */ info);

        // QR decomposition.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(geqrf)(
        //     int m, int n, scalar_t *a, int lda,
        //     scalar_t *tau, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_geqrf(
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Build Q from output of geqrf
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(orgqr)(
        //     int m, int n, int k, scalar_t *a, int lda,
        //     scalar_t *tau, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_orgqr(
            int m,
            int n,
            int k,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Multiply Q with a matrix from output of geqrf
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(ormqr)(
        //     char side, char trans, int m, int n, int k,
        //     scalar_t *a, int lda, scalar_t *tau, scalar_t *c, int ldc,
        //     scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THLongLapack_ormqr(
            byte side,
            byte trans,
            int m,
            int n,
            int k,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ c,
            int ldc,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Solve AX=B
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gesv)(
        //     int n, int nrhs, scalar_t *a, int lda, int *ipiv,
        //      scalar_t *b, int ldb, int* info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_gesv(
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Solve a triangular system of the form A * X = B or A^T * X = B
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(trtrs)(
        //     char uplo, char trans, char diag, int n, int nrhs,
        //     scalar_t *a, int lda, scalar_t *b, int ldb, int* info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_trtrs(
            byte uplo,
            byte trans,
            byte diag,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Evaluate ||AX-B||
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gels)(
        //     char trans, int m, int n, int nrhs, scalar_t *a, int lda,
        //     scalar_t *b, int ldb, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_gels(
            byte trans,
            int m,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Eigenvals
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(syev)(
        //     char jobz, char uplo, int n, scalar_t *a, int lda,
        //     scalar_t *w, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_syev(
            byte jobz,
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ w,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Non-sym eigenvals
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(geev)(
        //     char jobvl, char jobvr, int n, scalar_t *a, int lda,
        //     scalar_t *wr, scalar_t *wi, scalar_t* vl, int ldvl,
        //     scalar_t *vr, int ldvr, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_geev(
            byte jobvl,
            byte jobvr,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ wr,
            IntPtr /* scalar_t* */ wi,
            IntPtr /* scalar_t* */ vl,
            int ldvl,
            IntPtr /* scalar_t* */ vr,
            int ldvr,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // SVD
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gesdd)(
        //     char jobz, int m, int n, scalar_t *a, int lda, scalar_t *s,
        //     scalar_t *u, int ldu, scalar_t *vt, int ldvt,
        //     scalar_t *work, int lwork, int *iwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_gesdd(
            byte jobz,
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ s,
            IntPtr /* scalar_t* */ u,
            int ldu,
            IntPtr /* scalar_t* */ vt,
            int ldvt,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ iwork,
            IntPtr /* int* */ info);

        // LU decomposition
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getrf)(
        //     int m, int n, scalar_t *a, int lda, int *ipiv, int *info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_getrf(
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* int* */ info);

        // LU decomposition
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getrs)(
        //     char trans, int n, int nrhs, scalar_t *a, int lda,
        //     int *ipiv, scalar_t *b, int ldb, int *info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_getrs(
            byte trans,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Matrix Inverse
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getri)(
        //     int n, scalar_t *a, int lda, int *ipiv,
        //     scalar_t *work, int lwork, int* info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_getri(
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // *** Positive Definite matrices

        // Cholesky factorization
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potrf)(
        //     char uplo, int n, scalar_t *a, int lda, int *info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_potrf(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ info);

        // Matrix inverse based on Cholesky factorization
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potri)(
        //     char uplo, int n, scalar_t *a, int lda, int *info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_potri(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ info);

        // Solve A*X = B with a symmetric positive definite matrix A
        // using the Cholesky factorization.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potrs)(
        //     char uplo, int n, int nrhs, scalar_t *a, int lda,
        //     scalar_t *b, int ldb, int *info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_potrs(
            char uplo,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Cholesky factorization with complete pivoting.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(pstrf)(
        //     char uplo, int n, scalar_t *a, int lda, int *piv,
        //     int *rank, scalar_t tol, scalar_t *work, int *info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_pstrf(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ piv,
            IntPtr /* int* */ rank,
            double tol,
            IntPtr /* scalar_t* */ work,
            IntPtr /* int* */ info);

        // QR decomposition.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(geqrf)(
        //     int m, int n, scalar_t *a, int lda,
        //     scalar_t *tau, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_geqrf(
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Build Q from output of geqrf
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(orgqr)(
        //     int m, int n, int k, scalar_t *a, int lda,
        //     scalar_t *tau, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_orgqr(
            int m,
            int n,
            int k,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Multiply Q with a matrix from output of geqrf
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(ormqr)(
        //     char side, char trans, int m, int n, int k,
        //     scalar_t *a, int lda, scalar_t *tau, scalar_t *c, int ldc,
        //     scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THDoubleLapack_ormqr(
            byte side,
            byte trans,
            int m,
            int n,
            int k,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ c,
            int ldc,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Solve AX=B
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gesv)(
        //     int n, int nrhs, scalar_t *a, int lda, int *ipiv,
        //      scalar_t *b, int ldb, int* info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_gesv(
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Solve a triangular system of the form A * X = B or A^T * X = B
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(trtrs)(
        //     char uplo, char trans, char diag, int n, int nrhs,
        //     scalar_t *a, int lda, scalar_t *b, int ldb, int* info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_trtrs(
            byte uplo,
            byte trans,
            byte diag,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Evaluate ||AX-B||
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gels)(
        //     char trans, int m, int n, int nrhs, scalar_t *a, int lda,
        //     scalar_t *b, int ldb, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_gels(
            byte trans,
            int m,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Eigenvals
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(syev)(
        //     char jobz, char uplo, int n, scalar_t *a, int lda,
        //     scalar_t *w, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_syev(
            byte jobz,
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ w,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Non-sym eigenvals
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(geev)(
        //     char jobvl, char jobvr, int n, scalar_t *a, int lda,
        //     scalar_t *wr, scalar_t *wi, scalar_t* vl, int ldvl,
        //     scalar_t *vr, int ldvr, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_geev(
            byte jobvl,
            byte jobvr,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ wr,
            IntPtr /* scalar_t* */ wi,
            IntPtr /* scalar_t* */ vl,
            int ldvl,
            IntPtr /* scalar_t* */ vr,
            int ldvr,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // SVD
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(gesdd)(
        //     char jobz, int m, int n, scalar_t *a, int lda, scalar_t *s,
        //     scalar_t *u, int ldu, scalar_t *vt, int ldvt,
        //     scalar_t *work, int lwork, int *iwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_gesdd(
            byte jobz,
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ s,
            IntPtr /* scalar_t* */ u,
            int ldu,
            IntPtr /* scalar_t* */ vt,
            int ldvt,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ iwork,
            IntPtr /* int* */ info);

        // LU decomposition
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getrf)(
        //     int m, int n, scalar_t *a, int lda, int *ipiv, int *info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_getrf(
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* int* */ info);

        // LU decomposition
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getrs)(
        //     char trans, int n, int nrhs, scalar_t *a, int lda,
        //     int *ipiv, scalar_t *b, int ldb, int *info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_getrs(
            byte trans,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Matrix Inverse
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(getri)(
        //     int n, scalar_t *a, int lda, int *ipiv,
        //     scalar_t *work, int lwork, int* info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_getri(
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ ipiv,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // *** Positive Definite matrices

        // Cholesky factorization
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potrf)(
        //     char uplo, int n, scalar_t *a, int lda, int *info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_potrf(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ info);

        // Matrix inverse based on Cholesky factorization
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potri)(
        //     char uplo, int n, scalar_t *a, int lda, int *info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_potri(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ info);

        // Solve A*X = B with a symmetric positive definite matrix A
        // using the Cholesky factorization.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(potrs)(
        //     char uplo, int n, int nrhs, scalar_t *a, int lda,
        //     scalar_t *b, int ldb, int *info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_potrs(
            char uplo,
            int n,
            int nrhs,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ b,
            int ldb,
            IntPtr /* int* */ info);

        // Cholesky factorization with complete pivoting.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(pstrf)(
        //     char uplo, int n, scalar_t *a, int lda, int *piv,
        //     int *rank, scalar_t tol, scalar_t *work, int *info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_pstrf(
            byte uplo,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* int* */ piv,
            IntPtr /* int* */ rank,
            float tol,
            IntPtr /* scalar_t* */ work,
            IntPtr /* int* */ info);

        // QR decomposition.
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(geqrf)(
        //     int m, int n, scalar_t *a, int lda,
        //     scalar_t *tau, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_geqrf(
            int m,
            int n,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Build Q from output of geqrf
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(orgqr)(
        //     int m, int n, int k, scalar_t *a, int lda,
        //     scalar_t *tau, scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_orgqr(
            int m,
            int n,
            int k,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);

        // Multiply Q with a matrix from output of geqrf
        //
        // Corresponds to the following TH definition:
        //
        // TH_API void THLapack_(ormqr)(
        //     char side, char trans, int m, int n, int k,
        //     scalar_t *a, int lda, scalar_t *tau, scalar_t *c, int ldc,
        //     scalar_t *work, int lwork, int *info);
        [DllImport ("caffe2")]
        internal extern static void THFloatLapack_ormqr(
            byte side,
            byte trans,
            int m,
            int n,
            int k,
            IntPtr /* scalar_t* */ a,
            int lda,
            IntPtr /* scalar_t* */ tau,
            IntPtr /* scalar_t* */ c,
            int ldc,
            IntPtr /* scalar_t* */ work,
            int lwork,
            IntPtr /* int* */ info);
    }
}
