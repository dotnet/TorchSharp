using System;
using System.Buffers;
using System.Linq;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using TorchSharp;

namespace Torch.SNT 
{

    /// <summary>
    ///   Wrapper class used to surface a Torch ByteTensor as a System.Numerics DensorTensor of byte
    /// </summary>
    public sealed class ByteTorchTensor : DenseTensor<byte>, IDisposable
    {
        internal sealed class ByteNativeMemory : MemoryManager<byte>
        {
            private readonly ByteTensor.ByteStorage storage;

            public ByteNativeMemory(ByteTensor.ByteStorage storage)
            {
                this.storage = storage;

                if (storage.Size () < 0)
                {
                    throw new ArgumentOutOfRangeException ("Length cannot be negative.");
                }
            }

            /// <summary>
            /// Returns a span wrapping the underlying memory.
            /// Remember to Unpin the memory once the span is disposed.
            /// </summary>
            public override Span<byte> GetSpan ()
            {
                ulong len = storage.Size ();

                if (len > int.MaxValue)
                {
                    throw new InvalidCastException ("Tensor size not supported.");
                }

                unsafe
                {
                    return new Span<byte> (storage.Data.ToPointer (), (int)len);
                }
            }

            /// <summary>
            /// Returns a handle to the memory that has been pinned and hence its address can be taken.
            /// </summary>
            /// <param name="elementIndex">The offset to the element within the memory at which the returned <see cref="MemoryHandle"/> points to. (default = 0)</param>
            public override MemoryHandle Pin (int elementIndex = 0)
            {
                unsafe
                {
                    storage.Retain();
                    if ((uint)elementIndex > storage.Size ()) throw new ArgumentOutOfRangeException (nameof (elementIndex), "Index out of array bound.");
                    void* pointer = Unsafe.Add<byte> ((void*)storage.Data, elementIndex);
                    return new MemoryHandle (pointer, default, this);
                }
            }

            /// <summary>
            /// Lets the garbage collector know that the object is free to be moved now.
            /// </summary>
            public override void Unpin ()
            {
                storage.Free ();
            }

            protected override void Dispose (bool disposing)
            {
                if (disposing)
                {
                    storage.Free ();
                }
            }
        }

        private readonly object inner;

        /// <summary>
        ///   Property returning the inner TorchSharp tensor the class is wrapping. 
        /// </summary>
        public ByteTensor TorchSharpTensor => inner as ByteTensor;

        public ByteTorchTensor (Memory<byte> memory, ReadOnlySpan<int> dimensions, ByteTensor inner) : base (memory, dimensions)
        {
            this.inner = inner;
        }

        /// <summary>
        ///   Utility method to create a TorchTensor.
        ///   This is currently failing if the input parameter is empty because SNT 
        ///   does not support zero-size tensors.
        /// </summary>
        /// <param name="sizes">The desired sizes for the dimensions of the tensor.</param>
        public static ByteTorchTensor Create (params int[] sizes)
        {
            var totLength = Utils.GetTotalLength (sizes);
            var shape = sizes;

            if (sizes.Length == 0)
            {
                shape = new int[] { 0 };
            }

            var inner = CreateByteTensor (sizes.Select (x => (long)x).ToArray ());
            var storage = inner.Storage;
            storage.Retain ();
            var mem = new ByteNativeMemory (storage);

            return new ByteTorchTensor (mem.Memory, shape, inner);
        }

        public void Dispose ()
        {
            var typedInner = inner as ByteTensor;
            typedInner.Dispose ();
        }

        /// <summary>
        /// Creates a shallow copy of this tensor, with new backing storage.
        /// </summary>
        /// <returns>A shallow copy of this tensor.</returns>
        public override Tensor<byte> Clone ()
        {
            var typedInner = inner as ByteTensor;
            var innerClone = typedInner.Clone ();
            var storage = innerClone.Storage;
            storage.Retain ();
            var mem = new ByteNativeMemory (storage);

            return new ByteTorchTensor (mem.Memory, Dimensions, innerClone);
        }

        /// <summary>
        /// Creates a new Tensor of a different type with the specified dimensions and the same layout as this tensor with elements initialized to their default value.
        /// </summary>
        /// <typeparam name="TResult">Type contained in the returned Tensor.</typeparam>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor with the same layout as this tensor but different type and dimensions.</returns>
        public override Tensor<TResult> CloneEmpty<TResult> (ReadOnlySpan<int> dimensions)
        {
            var typedInner = inner as ByteTensor;

            switch (true)
            {
                case bool _ when typeof (TResult) == typeof (byte):
                    var innerClone = CreateByteTensor (typedInner.Shape);
                    innerClone.Fill (default);
                    var mem = new ByteNativeMemory (innerClone.Storage);

                    return new ByteTorchTensor (mem.Memory, Dimensions, innerClone) as Tensor<TResult>;
                default: throw new NotImplementedException ("Only cloning bytes is currently implemented.");
            }
        }

        /// <summary>
        /// Reshapes the current tensor to new dimensions, using the same backing storage.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor that reinterprets backing Buffer of this tensor with different dimensions.</returns>
        public override Tensor<byte> Reshape (ReadOnlySpan<int> dimensions)
        {
            if (dimensions.Length == 0)
            {
                throw new ArgumentException ("Dimensions must contain elements.", nameof (dimensions));
            }

            var newSize = Utils.GetTotalLength (dimensions);

            if (newSize != Length)
            {
                throw new ArgumentException ($"Cannot reshape array due to mismatch in lengths, currently {Length} would become {newSize}.", nameof(dimensions));
            }

            var typedInner = inner as ByteTensor;
            ByteTensor reshapedTensor;

            switch (dimensions.Length)
            {
                case 1:
                    reshapedTensor = typedInner.NewWithStorage1d (UIntPtr.Zero, dimensions[0], 1);
                    break;
                case 2:
                    reshapedTensor = typedInner.NewWithStorage2d (UIntPtr.Zero, dimensions[0], dimensions[1], dimensions[1], 1);
                    break;
                default: throw new ArgumentException ($"Cannot reshape tensor with more than 4 dimensions");
            }

            return new ByteTorchTensor (Buffer, dimensions, reshapedTensor);
        }

        /// <summary>
        ///   Creates a 1-4D tensor of the specified size(s).
        /// </summary>    
        /// <param name="dims">Sizes for the dimensions.</param>
        internal static ByteTensor CreateByteTensor (params long[] dims)
        {
            switch (dims.Length)
            {
                case 0:
                    return new ByteTensor ();
                case 1:
                    return new ByteTensor (dims[0]);
                case 2:
                    return new ByteTensor (dims[0], dims[1]);
                case 3:
                    return new ByteTensor (dims[0], dims[1], dims[2]);
                case 4:
                    return new ByteTensor (dims[0], dims[1], dims[2], dims[3]);
                default:
                    throw new ArgumentOutOfRangeException (nameof (dims), "Maximum number of dimensions for tensor creation is 4.");
            }
        }
    }

    /// <summary>
    ///   Wrapper class used to surface a Torch ShortTensor as a System.Numerics DensorTensor of short
    /// </summary>
    public sealed class ShortTorchTensor : DenseTensor<short>, IDisposable
    {
        internal sealed class ShortNativeMemory : MemoryManager<short>
        {
            private readonly ShortTensor.ShortStorage storage;

            public ShortNativeMemory(ShortTensor.ShortStorage storage)
            {
                this.storage = storage;

                if (storage.Size () < 0)
                {
                    throw new ArgumentOutOfRangeException ("Length cannot be negative.");
                }
            }

            /// <summary>
            /// Returns a span wrapping the underlying memory.
            /// Remember to Unpin the memory once the span is disposed.
            /// </summary>
            public override Span<short> GetSpan ()
            {
                ulong len = storage.Size ();

                if (len > int.MaxValue)
                {
                    throw new InvalidCastException ("Tensor size not supported.");
                }

                unsafe
                {
                    return new Span<short> (storage.Data.ToPointer (), (int)len);
                }
            }

            /// <summary>
            /// Returns a handle to the memory that has been pinned and hence its address can be taken.
            /// </summary>
            /// <param name="elementIndex">The offset to the element within the memory at which the returned <see cref="MemoryHandle"/> points to. (default = 0)</param>
            public override MemoryHandle Pin (int elementIndex = 0)
            {
                unsafe
                {
                    storage.Retain();
                    if ((uint)elementIndex > storage.Size ()) throw new ArgumentOutOfRangeException (nameof (elementIndex), "Index out of array bound.");
                    void* pointer = Unsafe.Add<short> ((void*)storage.Data, elementIndex);
                    return new MemoryHandle (pointer, default, this);
                }
            }

            /// <summary>
            /// Lets the garbage collector know that the object is free to be moved now.
            /// </summary>
            public override void Unpin ()
            {
                storage.Free ();
            }

            protected override void Dispose (bool disposing)
            {
                if (disposing)
                {
                    storage.Free ();
                }
            }
        }

        private readonly object inner;

        /// <summary>
        ///   Property returning the inner TorchSharp tensor the class is wrapping. 
        /// </summary>
        public ShortTensor TorchSharpTensor => inner as ShortTensor;

        public ShortTorchTensor (Memory<short> memory, ReadOnlySpan<int> dimensions, ShortTensor inner) : base (memory, dimensions)
        {
            this.inner = inner;
        }

        /// <summary>
        ///   Utility method to create a TorchTensor.
        ///   This is currently failing if the input parameter is empty because SNT 
        ///   does not support zero-size tensors.
        /// </summary>
        /// <param name="sizes">The desired sizes for the dimensions of the tensor.</param>
        public static ShortTorchTensor Create (params int[] sizes)
        {
            var totLength = Utils.GetTotalLength (sizes);
            var shape = sizes;

            if (sizes.Length == 0)
            {
                shape = new int[] { 0 };
            }

            var inner = CreateShortTensor (sizes.Select (x => (long)x).ToArray ());
            var storage = inner.Storage;
            storage.Retain ();
            var mem = new ShortNativeMemory (storage);

            return new ShortTorchTensor (mem.Memory, shape, inner);
        }

        public void Dispose ()
        {
            var typedInner = inner as ShortTensor;
            typedInner.Dispose ();
        }

        /// <summary>
        /// Creates a shallow copy of this tensor, with new backing storage.
        /// </summary>
        /// <returns>A shallow copy of this tensor.</returns>
        public override Tensor<short> Clone ()
        {
            var typedInner = inner as ShortTensor;
            var innerClone = typedInner.Clone ();
            var storage = innerClone.Storage;
            storage.Retain ();
            var mem = new ShortNativeMemory (storage);

            return new ShortTorchTensor (mem.Memory, Dimensions, innerClone);
        }

        /// <summary>
        /// Creates a new Tensor of a different type with the specified dimensions and the same layout as this tensor with elements initialized to their default value.
        /// </summary>
        /// <typeparam name="TResult">Type contained in the returned Tensor.</typeparam>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor with the same layout as this tensor but different type and dimensions.</returns>
        public override Tensor<TResult> CloneEmpty<TResult> (ReadOnlySpan<int> dimensions)
        {
            var typedInner = inner as ShortTensor;

            switch (true)
            {
                case bool _ when typeof (TResult) == typeof (short):
                    var innerClone = CreateShortTensor (typedInner.Shape);
                    innerClone.Fill (default);
                    var mem = new ShortNativeMemory (innerClone.Storage);

                    return new ShortTorchTensor (mem.Memory, Dimensions, innerClone) as Tensor<TResult>;
                default: throw new NotImplementedException ("Only cloning shorts is currently implemented.");
            }
        }

        /// <summary>
        /// Reshapes the current tensor to new dimensions, using the same backing storage.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor that reinterprets backing Buffer of this tensor with different dimensions.</returns>
        public override Tensor<short> Reshape (ReadOnlySpan<int> dimensions)
        {
            if (dimensions.Length == 0)
            {
                throw new ArgumentException ("Dimensions must contain elements.", nameof (dimensions));
            }

            var newSize = Utils.GetTotalLength (dimensions);

            if (newSize != Length)
            {
                throw new ArgumentException ($"Cannot reshape array due to mismatch in lengths, currently {Length} would become {newSize}.", nameof(dimensions));
            }

            var typedInner = inner as ShortTensor;
            ShortTensor reshapedTensor;

            switch (dimensions.Length)
            {
                case 1:
                    reshapedTensor = typedInner.NewWithStorage1d (UIntPtr.Zero, dimensions[0], 1);
                    break;
                case 2:
                    reshapedTensor = typedInner.NewWithStorage2d (UIntPtr.Zero, dimensions[0], dimensions[1], dimensions[1], 1);
                    break;
                default: throw new ArgumentException ($"Cannot reshape tensor with more than 4 dimensions");
            }

            return new ShortTorchTensor (Buffer, dimensions, reshapedTensor);
        }

        /// <summary>
        ///   Creates a 1-4D tensor of the specified size(s).
        /// </summary>    
        /// <param name="dims">Sizes for the dimensions.</param>
        internal static ShortTensor CreateShortTensor (params long[] dims)
        {
            switch (dims.Length)
            {
                case 0:
                    return new ShortTensor ();
                case 1:
                    return new ShortTensor (dims[0]);
                case 2:
                    return new ShortTensor (dims[0], dims[1]);
                case 3:
                    return new ShortTensor (dims[0], dims[1], dims[2]);
                case 4:
                    return new ShortTensor (dims[0], dims[1], dims[2], dims[3]);
                default:
                    throw new ArgumentOutOfRangeException (nameof (dims), "Maximum number of dimensions for tensor creation is 4.");
            }
        }
    }

    /// <summary>
    ///   Wrapper class used to surface a Torch IntTensor as a System.Numerics DensorTensor of int
    /// </summary>
    public sealed class IntTorchTensor : DenseTensor<int>, IDisposable
    {
        internal sealed class IntNativeMemory : MemoryManager<int>
        {
            private readonly IntTensor.IntStorage storage;

            public IntNativeMemory(IntTensor.IntStorage storage)
            {
                this.storage = storage;

                if (storage.Size () < 0)
                {
                    throw new ArgumentOutOfRangeException ("Length cannot be negative.");
                }
            }

            /// <summary>
            /// Returns a span wrapping the underlying memory.
            /// Remember to Unpin the memory once the span is disposed.
            /// </summary>
            public override Span<int> GetSpan ()
            {
                ulong len = storage.Size ();

                if (len > int.MaxValue)
                {
                    throw new InvalidCastException ("Tensor size not supported.");
                }

                unsafe
                {
                    return new Span<int> (storage.Data.ToPointer (), (int)len);
                }
            }

            /// <summary>
            /// Returns a handle to the memory that has been pinned and hence its address can be taken.
            /// </summary>
            /// <param name="elementIndex">The offset to the element within the memory at which the returned <see cref="MemoryHandle"/> points to. (default = 0)</param>
            public override MemoryHandle Pin (int elementIndex = 0)
            {
                unsafe
                {
                    storage.Retain();
                    if ((uint)elementIndex > storage.Size ()) throw new ArgumentOutOfRangeException (nameof (elementIndex), "Index out of array bound.");
                    void* pointer = Unsafe.Add<int> ((void*)storage.Data, elementIndex);
                    return new MemoryHandle (pointer, default, this);
                }
            }

            /// <summary>
            /// Lets the garbage collector know that the object is free to be moved now.
            /// </summary>
            public override void Unpin ()
            {
                storage.Free ();
            }

            protected override void Dispose (bool disposing)
            {
                if (disposing)
                {
                    storage.Free ();
                }
            }
        }

        private readonly object inner;

        /// <summary>
        ///   Property returning the inner TorchSharp tensor the class is wrapping. 
        /// </summary>
        public IntTensor TorchSharpTensor => inner as IntTensor;

        public IntTorchTensor (Memory<int> memory, ReadOnlySpan<int> dimensions, IntTensor inner) : base (memory, dimensions)
        {
            this.inner = inner;
        }

        /// <summary>
        ///   Utility method to create a TorchTensor.
        ///   This is currently failing if the input parameter is empty because SNT 
        ///   does not support zero-size tensors.
        /// </summary>
        /// <param name="sizes">The desired sizes for the dimensions of the tensor.</param>
        public static IntTorchTensor Create (params int[] sizes)
        {
            var totLength = Utils.GetTotalLength (sizes);
            var shape = sizes;

            if (sizes.Length == 0)
            {
                shape = new int[] { 0 };
            }

            var inner = CreateIntTensor (sizes.Select (x => (long)x).ToArray ());
            var storage = inner.Storage;
            storage.Retain ();
            var mem = new IntNativeMemory (storage);

            return new IntTorchTensor (mem.Memory, shape, inner);
        }

        public void Dispose ()
        {
            var typedInner = inner as IntTensor;
            typedInner.Dispose ();
        }

        /// <summary>
        /// Creates a shallow copy of this tensor, with new backing storage.
        /// </summary>
        /// <returns>A shallow copy of this tensor.</returns>
        public override Tensor<int> Clone ()
        {
            var typedInner = inner as IntTensor;
            var innerClone = typedInner.Clone ();
            var storage = innerClone.Storage;
            storage.Retain ();
            var mem = new IntNativeMemory (storage);

            return new IntTorchTensor (mem.Memory, Dimensions, innerClone);
        }

        /// <summary>
        /// Creates a new Tensor of a different type with the specified dimensions and the same layout as this tensor with elements initialized to their default value.
        /// </summary>
        /// <typeparam name="TResult">Type contained in the returned Tensor.</typeparam>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor with the same layout as this tensor but different type and dimensions.</returns>
        public override Tensor<TResult> CloneEmpty<TResult> (ReadOnlySpan<int> dimensions)
        {
            var typedInner = inner as IntTensor;

            switch (true)
            {
                case bool _ when typeof (TResult) == typeof (int):
                    var innerClone = CreateIntTensor (typedInner.Shape);
                    innerClone.Fill (default);
                    var mem = new IntNativeMemory (innerClone.Storage);

                    return new IntTorchTensor (mem.Memory, Dimensions, innerClone) as Tensor<TResult>;
                default: throw new NotImplementedException ("Only cloning ints is currently implemented.");
            }
        }

        /// <summary>
        /// Reshapes the current tensor to new dimensions, using the same backing storage.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor that reinterprets backing Buffer of this tensor with different dimensions.</returns>
        public override Tensor<int> Reshape (ReadOnlySpan<int> dimensions)
        {
            if (dimensions.Length == 0)
            {
                throw new ArgumentException ("Dimensions must contain elements.", nameof (dimensions));
            }

            var newSize = Utils.GetTotalLength (dimensions);

            if (newSize != Length)
            {
                throw new ArgumentException ($"Cannot reshape array due to mismatch in lengths, currently {Length} would become {newSize}.", nameof(dimensions));
            }

            var typedInner = inner as IntTensor;
            IntTensor reshapedTensor;

            switch (dimensions.Length)
            {
                case 1:
                    reshapedTensor = typedInner.NewWithStorage1d (UIntPtr.Zero, dimensions[0], 1);
                    break;
                case 2:
                    reshapedTensor = typedInner.NewWithStorage2d (UIntPtr.Zero, dimensions[0], dimensions[1], dimensions[1], 1);
                    break;
                default: throw new ArgumentException ($"Cannot reshape tensor with more than 4 dimensions");
            }

            return new IntTorchTensor (Buffer, dimensions, reshapedTensor);
        }

        /// <summary>
        ///   Creates a 1-4D tensor of the specified size(s).
        /// </summary>    
        /// <param name="dims">Sizes for the dimensions.</param>
        internal static IntTensor CreateIntTensor (params long[] dims)
        {
            switch (dims.Length)
            {
                case 0:
                    return new IntTensor ();
                case 1:
                    return new IntTensor (dims[0]);
                case 2:
                    return new IntTensor (dims[0], dims[1]);
                case 3:
                    return new IntTensor (dims[0], dims[1], dims[2]);
                case 4:
                    return new IntTensor (dims[0], dims[1], dims[2], dims[3]);
                default:
                    throw new ArgumentOutOfRangeException (nameof (dims), "Maximum number of dimensions for tensor creation is 4.");
            }
        }
    }

    /// <summary>
    ///   Wrapper class used to surface a Torch LongTensor as a System.Numerics DensorTensor of long
    /// </summary>
    public sealed class LongTorchTensor : DenseTensor<long>, IDisposable
    {
        internal sealed class LongNativeMemory : MemoryManager<long>
        {
            private readonly LongTensor.LongStorage storage;

            public LongNativeMemory(LongTensor.LongStorage storage)
            {
                this.storage = storage;

                if (storage.Size () < 0)
                {
                    throw new ArgumentOutOfRangeException ("Length cannot be negative.");
                }
            }

            /// <summary>
            /// Returns a span wrapping the underlying memory.
            /// Remember to Unpin the memory once the span is disposed.
            /// </summary>
            public override Span<long> GetSpan ()
            {
                ulong len = storage.Size ();

                if (len > int.MaxValue)
                {
                    throw new InvalidCastException ("Tensor size not supported.");
                }

                unsafe
                {
                    return new Span<long> (storage.Data.ToPointer (), (int)len);
                }
            }

            /// <summary>
            /// Returns a handle to the memory that has been pinned and hence its address can be taken.
            /// </summary>
            /// <param name="elementIndex">The offset to the element within the memory at which the returned <see cref="MemoryHandle"/> points to. (default = 0)</param>
            public override MemoryHandle Pin (int elementIndex = 0)
            {
                unsafe
                {
                    storage.Retain();
                    if ((uint)elementIndex > storage.Size ()) throw new ArgumentOutOfRangeException (nameof (elementIndex), "Index out of array bound.");
                    void* pointer = Unsafe.Add<long> ((void*)storage.Data, elementIndex);
                    return new MemoryHandle (pointer, default, this);
                }
            }

            /// <summary>
            /// Lets the garbage collector know that the object is free to be moved now.
            /// </summary>
            public override void Unpin ()
            {
                storage.Free ();
            }

            protected override void Dispose (bool disposing)
            {
                if (disposing)
                {
                    storage.Free ();
                }
            }
        }

        private readonly object inner;

        /// <summary>
        ///   Property returning the inner TorchSharp tensor the class is wrapping. 
        /// </summary>
        public LongTensor TorchSharpTensor => inner as LongTensor;

        public LongTorchTensor (Memory<long> memory, ReadOnlySpan<int> dimensions, LongTensor inner) : base (memory, dimensions)
        {
            this.inner = inner;
        }

        /// <summary>
        ///   Utility method to create a TorchTensor.
        ///   This is currently failing if the input parameter is empty because SNT 
        ///   does not support zero-size tensors.
        /// </summary>
        /// <param name="sizes">The desired sizes for the dimensions of the tensor.</param>
        public static LongTorchTensor Create (params int[] sizes)
        {
            var totLength = Utils.GetTotalLength (sizes);
            var shape = sizes;

            if (sizes.Length == 0)
            {
                shape = new int[] { 0 };
            }

            var inner = CreateLongTensor (sizes.Select (x => (long)x).ToArray ());
            var storage = inner.Storage;
            storage.Retain ();
            var mem = new LongNativeMemory (storage);

            return new LongTorchTensor (mem.Memory, shape, inner);
        }

        public void Dispose ()
        {
            var typedInner = inner as LongTensor;
            typedInner.Dispose ();
        }

        /// <summary>
        /// Creates a shallow copy of this tensor, with new backing storage.
        /// </summary>
        /// <returns>A shallow copy of this tensor.</returns>
        public override Tensor<long> Clone ()
        {
            var typedInner = inner as LongTensor;
            var innerClone = typedInner.Clone ();
            var storage = innerClone.Storage;
            storage.Retain ();
            var mem = new LongNativeMemory (storage);

            return new LongTorchTensor (mem.Memory, Dimensions, innerClone);
        }

        /// <summary>
        /// Creates a new Tensor of a different type with the specified dimensions and the same layout as this tensor with elements initialized to their default value.
        /// </summary>
        /// <typeparam name="TResult">Type contained in the returned Tensor.</typeparam>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor with the same layout as this tensor but different type and dimensions.</returns>
        public override Tensor<TResult> CloneEmpty<TResult> (ReadOnlySpan<int> dimensions)
        {
            var typedInner = inner as LongTensor;

            switch (true)
            {
                case bool _ when typeof (TResult) == typeof (long):
                    var innerClone = CreateLongTensor (typedInner.Shape);
                    innerClone.Fill (default);
                    var mem = new LongNativeMemory (innerClone.Storage);

                    return new LongTorchTensor (mem.Memory, Dimensions, innerClone) as Tensor<TResult>;
                default: throw new NotImplementedException ("Only cloning longs is currently implemented.");
            }
        }

        /// <summary>
        /// Reshapes the current tensor to new dimensions, using the same backing storage.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor that reinterprets backing Buffer of this tensor with different dimensions.</returns>
        public override Tensor<long> Reshape (ReadOnlySpan<int> dimensions)
        {
            if (dimensions.Length == 0)
            {
                throw new ArgumentException ("Dimensions must contain elements.", nameof (dimensions));
            }

            var newSize = Utils.GetTotalLength (dimensions);

            if (newSize != Length)
            {
                throw new ArgumentException ($"Cannot reshape array due to mismatch in lengths, currently {Length} would become {newSize}.", nameof(dimensions));
            }

            var typedInner = inner as LongTensor;
            LongTensor reshapedTensor;

            switch (dimensions.Length)
            {
                case 1:
                    reshapedTensor = typedInner.NewWithStorage1d (UIntPtr.Zero, dimensions[0], 1);
                    break;
                case 2:
                    reshapedTensor = typedInner.NewWithStorage2d (UIntPtr.Zero, dimensions[0], dimensions[1], dimensions[1], 1);
                    break;
                default: throw new ArgumentException ($"Cannot reshape tensor with more than 4 dimensions");
            }

            return new LongTorchTensor (Buffer, dimensions, reshapedTensor);
        }

        /// <summary>
        ///   Creates a 1-4D tensor of the specified size(s).
        /// </summary>    
        /// <param name="dims">Sizes for the dimensions.</param>
        internal static LongTensor CreateLongTensor (params long[] dims)
        {
            switch (dims.Length)
            {
                case 0:
                    return new LongTensor ();
                case 1:
                    return new LongTensor (dims[0]);
                case 2:
                    return new LongTensor (dims[0], dims[1]);
                case 3:
                    return new LongTensor (dims[0], dims[1], dims[2]);
                case 4:
                    return new LongTensor (dims[0], dims[1], dims[2], dims[3]);
                default:
                    throw new ArgumentOutOfRangeException (nameof (dims), "Maximum number of dimensions for tensor creation is 4.");
            }
        }
    }

    /// <summary>
    ///   Wrapper class used to surface a Torch DoubleTensor as a System.Numerics DensorTensor of double
    /// </summary>
    public sealed class DoubleTorchTensor : DenseTensor<double>, IDisposable
    {
        internal sealed class DoubleNativeMemory : MemoryManager<double>
        {
            private readonly DoubleTensor.DoubleStorage storage;

            public DoubleNativeMemory(DoubleTensor.DoubleStorage storage)
            {
                this.storage = storage;

                if (storage.Size () < 0)
                {
                    throw new ArgumentOutOfRangeException ("Length cannot be negative.");
                }
            }

            /// <summary>
            /// Returns a span wrapping the underlying memory.
            /// Remember to Unpin the memory once the span is disposed.
            /// </summary>
            public override Span<double> GetSpan ()
            {
                ulong len = storage.Size ();

                if (len > int.MaxValue)
                {
                    throw new InvalidCastException ("Tensor size not supported.");
                }

                unsafe
                {
                    return new Span<double> (storage.Data.ToPointer (), (int)len);
                }
            }

            /// <summary>
            /// Returns a handle to the memory that has been pinned and hence its address can be taken.
            /// </summary>
            /// <param name="elementIndex">The offset to the element within the memory at which the returned <see cref="MemoryHandle"/> points to. (default = 0)</param>
            public override MemoryHandle Pin (int elementIndex = 0)
            {
                unsafe
                {
                    storage.Retain();
                    if ((uint)elementIndex > storage.Size ()) throw new ArgumentOutOfRangeException (nameof (elementIndex), "Index out of array bound.");
                    void* pointer = Unsafe.Add<double> ((void*)storage.Data, elementIndex);
                    return new MemoryHandle (pointer, default, this);
                }
            }

            /// <summary>
            /// Lets the garbage collector know that the object is free to be moved now.
            /// </summary>
            public override void Unpin ()
            {
                storage.Free ();
            }

            protected override void Dispose (bool disposing)
            {
                if (disposing)
                {
                    storage.Free ();
                }
            }
        }

        private readonly object inner;

        /// <summary>
        ///   Property returning the inner TorchSharp tensor the class is wrapping. 
        /// </summary>
        public DoubleTensor TorchSharpTensor => inner as DoubleTensor;

        public DoubleTorchTensor (Memory<double> memory, ReadOnlySpan<int> dimensions, DoubleTensor inner) : base (memory, dimensions)
        {
            this.inner = inner;
        }

        /// <summary>
        ///   Utility method to create a TorchTensor.
        ///   This is currently failing if the input parameter is empty because SNT 
        ///   does not support zero-size tensors.
        /// </summary>
        /// <param name="sizes">The desired sizes for the dimensions of the tensor.</param>
        public static DoubleTorchTensor Create (params int[] sizes)
        {
            var totLength = Utils.GetTotalLength (sizes);
            var shape = sizes;

            if (sizes.Length == 0)
            {
                shape = new int[] { 0 };
            }

            var inner = CreateDoubleTensor (sizes.Select (x => (long)x).ToArray ());
            var storage = inner.Storage;
            storage.Retain ();
            var mem = new DoubleNativeMemory (storage);

            return new DoubleTorchTensor (mem.Memory, shape, inner);
        }

        public void Dispose ()
        {
            var typedInner = inner as DoubleTensor;
            typedInner.Dispose ();
        }

        /// <summary>
        /// Creates a shallow copy of this tensor, with new backing storage.
        /// </summary>
        /// <returns>A shallow copy of this tensor.</returns>
        public override Tensor<double> Clone ()
        {
            var typedInner = inner as DoubleTensor;
            var innerClone = typedInner.Clone ();
            var storage = innerClone.Storage;
            storage.Retain ();
            var mem = new DoubleNativeMemory (storage);

            return new DoubleTorchTensor (mem.Memory, Dimensions, innerClone);
        }

        /// <summary>
        /// Creates a new Tensor of a different type with the specified dimensions and the same layout as this tensor with elements initialized to their default value.
        /// </summary>
        /// <typeparam name="TResult">Type contained in the returned Tensor.</typeparam>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor with the same layout as this tensor but different type and dimensions.</returns>
        public override Tensor<TResult> CloneEmpty<TResult> (ReadOnlySpan<int> dimensions)
        {
            var typedInner = inner as DoubleTensor;

            switch (true)
            {
                case bool _ when typeof (TResult) == typeof (double):
                    var innerClone = CreateDoubleTensor (typedInner.Shape);
                    innerClone.Fill (default);
                    var mem = new DoubleNativeMemory (innerClone.Storage);

                    return new DoubleTorchTensor (mem.Memory, Dimensions, innerClone) as Tensor<TResult>;
                default: throw new NotImplementedException ("Only cloning doubles is currently implemented.");
            }
        }

        /// <summary>
        /// Reshapes the current tensor to new dimensions, using the same backing storage.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor that reinterprets backing Buffer of this tensor with different dimensions.</returns>
        public override Tensor<double> Reshape (ReadOnlySpan<int> dimensions)
        {
            if (dimensions.Length == 0)
            {
                throw new ArgumentException ("Dimensions must contain elements.", nameof (dimensions));
            }

            var newSize = Utils.GetTotalLength (dimensions);

            if (newSize != Length)
            {
                throw new ArgumentException ($"Cannot reshape array due to mismatch in lengths, currently {Length} would become {newSize}.", nameof(dimensions));
            }

            var typedInner = inner as DoubleTensor;
            DoubleTensor reshapedTensor;

            switch (dimensions.Length)
            {
                case 1:
                    reshapedTensor = typedInner.NewWithStorage1d (UIntPtr.Zero, dimensions[0], 1);
                    break;
                case 2:
                    reshapedTensor = typedInner.NewWithStorage2d (UIntPtr.Zero, dimensions[0], dimensions[1], dimensions[1], 1);
                    break;
                default: throw new ArgumentException ($"Cannot reshape tensor with more than 4 dimensions");
            }

            return new DoubleTorchTensor (Buffer, dimensions, reshapedTensor);
        }

        /// <summary>
        ///   Creates a 1-4D tensor of the specified size(s).
        /// </summary>    
        /// <param name="dims">Sizes for the dimensions.</param>
        internal static DoubleTensor CreateDoubleTensor (params long[] dims)
        {
            switch (dims.Length)
            {
                case 0:
                    return new DoubleTensor ();
                case 1:
                    return new DoubleTensor (dims[0]);
                case 2:
                    return new DoubleTensor (dims[0], dims[1]);
                case 3:
                    return new DoubleTensor (dims[0], dims[1], dims[2]);
                case 4:
                    return new DoubleTensor (dims[0], dims[1], dims[2], dims[3]);
                default:
                    throw new ArgumentOutOfRangeException (nameof (dims), "Maximum number of dimensions for tensor creation is 4.");
            }
        }
    }

    /// <summary>
    ///   Wrapper class used to surface a Torch FloatTensor as a System.Numerics DensorTensor of float
    /// </summary>
    public sealed class FloatTorchTensor : DenseTensor<float>, IDisposable
    {
        internal sealed class FloatNativeMemory : MemoryManager<float>
        {
            private readonly FloatTensor.FloatStorage storage;

            public FloatNativeMemory(FloatTensor.FloatStorage storage)
            {
                this.storage = storage;

                if (storage.Size () < 0)
                {
                    throw new ArgumentOutOfRangeException ("Length cannot be negative.");
                }
            }

            /// <summary>
            /// Returns a span wrapping the underlying memory.
            /// Remember to Unpin the memory once the span is disposed.
            /// </summary>
            public override Span<float> GetSpan ()
            {
                ulong len = storage.Size ();

                if (len > int.MaxValue)
                {
                    throw new InvalidCastException ("Tensor size not supported.");
                }

                unsafe
                {
                    return new Span<float> (storage.Data.ToPointer (), (int)len);
                }
            }

            /// <summary>
            /// Returns a handle to the memory that has been pinned and hence its address can be taken.
            /// </summary>
            /// <param name="elementIndex">The offset to the element within the memory at which the returned <see cref="MemoryHandle"/> points to. (default = 0)</param>
            public override MemoryHandle Pin (int elementIndex = 0)
            {
                unsafe
                {
                    storage.Retain();
                    if ((uint)elementIndex > storage.Size ()) throw new ArgumentOutOfRangeException (nameof (elementIndex), "Index out of array bound.");
                    void* pointer = Unsafe.Add<float> ((void*)storage.Data, elementIndex);
                    return new MemoryHandle (pointer, default, this);
                }
            }

            /// <summary>
            /// Lets the garbage collector know that the object is free to be moved now.
            /// </summary>
            public override void Unpin ()
            {
                storage.Free ();
            }

            protected override void Dispose (bool disposing)
            {
                if (disposing)
                {
                    storage.Free ();
                }
            }
        }

        private readonly object inner;

        /// <summary>
        ///   Property returning the inner TorchSharp tensor the class is wrapping. 
        /// </summary>
        public FloatTensor TorchSharpTensor => inner as FloatTensor;

        public FloatTorchTensor (Memory<float> memory, ReadOnlySpan<int> dimensions, FloatTensor inner) : base (memory, dimensions)
        {
            this.inner = inner;
        }

        /// <summary>
        ///   Utility method to create a TorchTensor.
        ///   This is currently failing if the input parameter is empty because SNT 
        ///   does not support zero-size tensors.
        /// </summary>
        /// <param name="sizes">The desired sizes for the dimensions of the tensor.</param>
        public static FloatTorchTensor Create (params int[] sizes)
        {
            var totLength = Utils.GetTotalLength (sizes);
            var shape = sizes;

            if (sizes.Length == 0)
            {
                shape = new int[] { 0 };
            }

            var inner = CreateFloatTensor (sizes.Select (x => (long)x).ToArray ());
            var storage = inner.Storage;
            storage.Retain ();
            var mem = new FloatNativeMemory (storage);

            return new FloatTorchTensor (mem.Memory, shape, inner);
        }

        public void Dispose ()
        {
            var typedInner = inner as FloatTensor;
            typedInner.Dispose ();
        }

        /// <summary>
        /// Creates a shallow copy of this tensor, with new backing storage.
        /// </summary>
        /// <returns>A shallow copy of this tensor.</returns>
        public override Tensor<float> Clone ()
        {
            var typedInner = inner as FloatTensor;
            var innerClone = typedInner.Clone ();
            var storage = innerClone.Storage;
            storage.Retain ();
            var mem = new FloatNativeMemory (storage);

            return new FloatTorchTensor (mem.Memory, Dimensions, innerClone);
        }

        /// <summary>
        /// Creates a new Tensor of a different type with the specified dimensions and the same layout as this tensor with elements initialized to their default value.
        /// </summary>
        /// <typeparam name="TResult">Type contained in the returned Tensor.</typeparam>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor with the same layout as this tensor but different type and dimensions.</returns>
        public override Tensor<TResult> CloneEmpty<TResult> (ReadOnlySpan<int> dimensions)
        {
            var typedInner = inner as FloatTensor;

            switch (true)
            {
                case bool _ when typeof (TResult) == typeof (float):
                    var innerClone = CreateFloatTensor (typedInner.Shape);
                    innerClone.Fill (default);
                    var mem = new FloatNativeMemory (innerClone.Storage);

                    return new FloatTorchTensor (mem.Memory, Dimensions, innerClone) as Tensor<TResult>;
                default: throw new NotImplementedException ("Only cloning floats is currently implemented.");
            }
        }

        /// <summary>
        /// Reshapes the current tensor to new dimensions, using the same backing storage.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor that reinterprets backing Buffer of this tensor with different dimensions.</returns>
        public override Tensor<float> Reshape (ReadOnlySpan<int> dimensions)
        {
            if (dimensions.Length == 0)
            {
                throw new ArgumentException ("Dimensions must contain elements.", nameof (dimensions));
            }

            var newSize = Utils.GetTotalLength (dimensions);

            if (newSize != Length)
            {
                throw new ArgumentException ($"Cannot reshape array due to mismatch in lengths, currently {Length} would become {newSize}.", nameof(dimensions));
            }

            var typedInner = inner as FloatTensor;
            FloatTensor reshapedTensor;

            switch (dimensions.Length)
            {
                case 1:
                    reshapedTensor = typedInner.NewWithStorage1d (UIntPtr.Zero, dimensions[0], 1);
                    break;
                case 2:
                    reshapedTensor = typedInner.NewWithStorage2d (UIntPtr.Zero, dimensions[0], dimensions[1], dimensions[1], 1);
                    break;
                default: throw new ArgumentException ($"Cannot reshape tensor with more than 4 dimensions");
            }

            return new FloatTorchTensor (Buffer, dimensions, reshapedTensor);
        }

        /// <summary>
        ///   Creates a 1-4D tensor of the specified size(s).
        /// </summary>    
        /// <param name="dims">Sizes for the dimensions.</param>
        internal static FloatTensor CreateFloatTensor (params long[] dims)
        {
            switch (dims.Length)
            {
                case 0:
                    return new FloatTensor ();
                case 1:
                    return new FloatTensor (dims[0]);
                case 2:
                    return new FloatTensor (dims[0], dims[1]);
                case 3:
                    return new FloatTensor (dims[0], dims[1], dims[2]);
                case 4:
                    return new FloatTensor (dims[0], dims[1], dims[2], dims[3]);
                default:
                    throw new ArgumentOutOfRangeException (nameof (dims), "Maximum number of dimensions for tensor creation is 4.");
            }
        }
    }
}
