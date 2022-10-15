// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class nn
        {
            public static partial class utils
            {
                public static partial class rnn
                {
                    /// <summary>
                    /// A packed batch of variable length sequences.
                    /// </summary>
                    public sealed class PackedSequence : IDisposable
                    {
                        /// <summary>
                        /// Class wrapping PyTorch's packedsequence object reference.
                        /// </summary>
                        internal sealed class HType : SafeHandle
                        {
                            public HType(IntPtr preexistingHandle, bool ownsHandle)
                                : base(IntPtr.Zero, ownsHandle)
                            {
                                SetHandle(preexistingHandle);
                            }

                            public override bool IsInvalid => handle == IntPtr.Zero;

                            // This is just for marshalling
                            internal HType() : base(IntPtr.Zero, true)
                            {
                            }

                            protected override bool ReleaseHandle()
                            {
                                THSNN_PackedSequence_dispose(handle);
                                return true;
                            }
                        }

                        /// <summary>
                        /// The packed sequences
                        /// </summary>
                        public readonly Tensor data;

                        /// <summary>
                        /// Batch size at each sequence step
                        /// </summary>
                        public readonly Tensor batch_sizes;

                        /// <summary>
                        /// The sorted indices
                        /// </summary>
                        public readonly Tensor sorted_indices;

                        /// <summary>
                        /// The original indices
                        /// </summary>
                        public readonly Tensor unsorted_indices;
                        private HType handle;

                        internal PackedSequence(HType handle)
                        {
                            this.handle = handle;
                            this.data = new Tensor(THSNN_PackedSequence_data(handle));
                            this.batch_sizes = new Tensor(THSNN_PackedSequence_batch_sizes(handle));
                            this.sorted_indices = new Tensor(THSNN_PackedSequence_sorted_indices(handle));
                            this.unsorted_indices = new Tensor(THSNN_PackedSequence_unsorted_indices(handle));
                        }

                        internal HType Handle => handle;

                        /// <summary>
                        ///   Releases the storage.
                        /// </summary>
                        public void Dispose()
                        {
                            this.data.Dispose();
                            this.batch_sizes.Dispose();
                            this.sorted_indices.Dispose();
                            this.unsorted_indices.Dispose();

                            if (handle != null && !handle.IsInvalid) {
                                handle.Dispose();
                                handle.SetHandleAsInvalid();
                            }
                        }
}
                }
            }
        }
    }
}
