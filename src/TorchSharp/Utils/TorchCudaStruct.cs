using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
namespace TorchSharp.Utils
{
#pragma warning disable 0169
    public struct cudaDeviceProp
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
        char[] name;                  /*< ASCII string identifying device */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
        char[] uuid;                       /*< 16-byte unique identifier */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        char[] luid;                    /*< 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms */
        uint luidDeviceNodeMask;         /*< LUID device node mask. Value is undefined on TCC and non-Windows platforms */
        ulong totalGlobalMem;             /*< Global memory available on device in bytes */
        ulong sharedMemPerBlock;          /*< Shared memory available per block in bytes */
        int regsPerBlock;               /*< 32-bit registers available per block */
        int warpSize;                   /*< Warp size in threads */
        ulong memPitch;                   /*< Maximum pitch in bytes allowed by memory copies */
        int maxThreadsPerBlock;         /*< Maximum number of threads per block */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        int[] maxThreadsDim;           /*< Maximum size of each dimension of a block */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        int[] maxGridSize;             /*< Maximum size of each dimension of a grid */
        int clockRate;                  /*< Deprecated, Clock frequency in kilohertz */
        ulong totalConstMem;              /*< Constant memory available on device in bytes */
        int major;                      /*< Major compute capability */
        int minor;                      /*< Minor compute capability */
        ulong textureAlignment;           /*< Alignment requirement for textures */
        ulong texturePitchAlignment;      /*< Pitch alignment requirement for texture references bound to pitched memory */
        int deviceOverlap;              /*< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
        int multiProcessorCount;        /*< Number of multiprocessors on device */
        int kernelExecTimeoutEnabled;   /*< Deprecated, Specified whether there is a run time limit on kernels */
        int integrated;                 /*< Device is integrated as opposed to discrete */
        int canMapHostMemory;           /*< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
        int computeMode;                /*< Deprecated, Compute mode (See ::cudaComputeMode) */
        int maxTexture1D;               /*< Maximum 1D texture size */
        int maxTexture1DMipmap;         /*< Maximum 1D mipmapped texture size */
        int maxTexture1DLinear;         /*< Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead. */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=2)]
        int[] maxTexture2D;            /*< Maximum 2D texture dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=2)]
        int[] maxTexture2DMipmap;      /*< Maximum 2D mipmapped texture dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=3)]
        int[] maxTexture2DLinear;      /*< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=2)]
        int[] maxTexture2DGather;      /*< Maximum 2D texture dimensions if texture gather operations have to be performed */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=3)]
        int[] maxTexture3D;            /*< Maximum 3D texture dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=3)]
        int[] maxTexture3DAlt;         /*< Maximum alternate 3D texture dimensions */
        int maxTextureCubemap;          /*< Maximum Cubemap texture dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=2)]
        int[] maxTexture1DLayered;     /*< Maximum 1D layered texture dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=3)]
        int[] maxTexture2DLayered;     /*< Maximum 2D layered texture dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=2)]
        int[] maxTextureCubemapLayered;/*< Maximum Cubemap layered texture dimensions */
        int maxSurface1D;               /*< Maximum 1D surface size */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=2)]
        int[] maxSurface2D;            /*< Maximum 2D surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=3)]
        int[] maxSurface3D;            /*< Maximum 3D surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=2)]
        int[] maxSurface1DLayered;     /*< Maximum 1D layered surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=3)]
        int[] maxSurface2DLayered;     /*< Maximum 2D layered surface dimensions */
        int maxSurfaceCubemap;          /*< Maximum Cubemap surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=2)]
        int[] maxSurfaceCubemapLayered;/*< Maximum Cubemap layered surface dimensions */
        ulong surfaceAlignment;           /*< Alignment requirements for surfaces */
        int concurrentKernels;          /*< Device can possibly execute multiple kernels concurrently */
        int ECCEnabled;                 /*< Device has ECC support enabled */
        int pciBusID;                   /*< PCI bus ID of the device */
        int pciDeviceID;                /*< PCI device ID of the device */
        int pciDomainID;                /*< PCI domain ID of the device */
        int tccDriver;                  /*< 1 if device is a Tesla device using TCC driver, 0 otherwise */
        int asyncEngineCount;           /*< Number of asynchronous engines */
        int unifiedAddressing;          /*< Device shares a unified address space with the host */
        int memoryClockRate;            /*< Deprecated, Peak memory clock frequency in kilohertz */
        int memoryBusWidth;             /*< Global memory bus width in bits */
        int l2CacheSize;                /*< Size of L2 cache in bytes */
        int persistingL2CacheMaxSize;   /*< Device's maximum l2 persisting lines capacity setting in bytes */
        int maxThreadsPerMultiProcessor;/*< Maximum resident threads per multiprocessor */
        int streamPrioritiesSupported;  /*< Device supports stream priorities */
        int globalL1CacheSupported;     /*< Device supports caching globals in L1 */
        int localL1CacheSupported;      /*< Device supports caching locals in L1 */
        ulong sharedMemPerMultiprocessor; /*< Shared memory available per multiprocessor in bytes */
        int regsPerMultiprocessor;      /*< 32-bit registers available per multiprocessor */
        int managedMemory;              /*< Device supports allocating managed memory on this system */
        int isMultiGpuBoard;            /*< Device is on a multi-GPU board */
        int multiGpuBoardGroupID;       /*< Unique identifier for a group of devices on the same multi-GPU board */
        int hostNativeAtomicSupported;  /*< Link between the device and the host supports native atomic operations */
        int singleToDoublePrecisionPerfRatio; /*< Deprecated, Ratio of single precision performance (in floating-point operations per second) to double precision performance */
        int pageableMemoryAccess;       /*< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
        int concurrentManagedAccess;    /*< Device can coherently access managed memory concurrently with the CPU */
        int computePreemptionSupported; /*< Device supports Compute Preemption */
        int canUseHostPointerForRegisteredMem; /*< Device can access host registered memory at the same virtual address as the CPU */
        int cooperativeLaunch;          /*< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel */
        int cooperativeMultiDeviceLaunch; /*< Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated. */
        ulong sharedMemPerBlockOptin;     /*< Per device maximum shared memory per block usable by special opt in */
        int pageableMemoryAccessUsesHostPageTables; /*< Device accesses pageable memory via the host's page tables */
        int directManagedMemAccessFromHost; /*< Host can directly access managed memory on the device without migration. */
        int maxBlocksPerMultiProcessor; /*< Maximum number of resident blocks per multiprocessor */
        int accessPolicyMaxWindowSize;  /*< The maximum value of ::cudaAccessPolicyWindow::num_bytes. */
        ulong reservedSharedMemPerBlock;  /*< Shared memory reserved by CUDA driver per block in bytes */
        int hostRegisterSupported;      /*< Device supports host memory registration via ::cudaHostRegister. */
        int sparseCudaArraySupported;   /*< 1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise */
        int hostRegisterReadOnlySupported; /*< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
        int timelineSemaphoreInteropSupported; /*< External timeline semaphore interop is supported on the device */
        int memoryPoolsSupported;       /*< 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise */
        int gpuDirectRDMASupported;     /*< 1 if the device supports GPUDirect RDMA APIs, 0 otherwise */
        uint gpuDirectRDMAFlushWritesOptions; /*< Bitmask to be interpreted according to the ::cudaFlushGPUDirectRDMAWritesOptions enum */
        int gpuDirectRDMAWritesOrdering;/*< See the ::cudaGPUDirectRDMAWritesOrdering enum for numerical values */
        uint memoryPoolSupportedHandleTypes; /*< Bitmask of handle types supported with mempool-based IPC */
        int deferredMappingCudaArraySupported; /*< 1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
        int ipcEventSupported;          /*< Device supports IPC Events. */
        int clusterLaunch;              /*< Indicates device supports cluster launch */
        int unifiedFunctionPointers;    /*< Indicates device supports unified pointers */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=2)]
        int[] reserved2;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=1)]
        int[] reserved1;               /*< Reserved for future use */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=60)]
        int[] reserved;               /*< Reserved for future use */
    }
#pragma warning restore 0169

}

