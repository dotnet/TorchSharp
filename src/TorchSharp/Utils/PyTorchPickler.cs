using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ICSharpCode.SharpZipLib.Zip.Compression.Streams;
using Razorvine.Pickle;
using Razorvine.Pickle.Objects;

namespace TorchSharp.Utils
{
    static class PyTorchPickler 
    {
        static PyTorchPickler()
        {
            Pickler.registerCustomPickler(typeof(Storage), new StoragePickler());
            Pickler.registerCustomDeconstructor(typeof(EmptyOrderedDict), new EmptyOrderedDictDeconstructor());
            Pickler.registerCustomDeconstructor(typeof(TensorWrapper), new TensorWrapperDeconstructor());
        }

        static byte MinProducedFileFormatVersion = 0x3;
        /// <summary>
        /// Pickle the state_dict to a python compatible file to be loaded using `torch.load`
        /// </summary>
        /// <param name="file">Path to the file</param>
        /// <param name="source">The state_dict to pickle</param>
        public static void PickleStateDict(string file, Dictionary<string, torch.Tensor> source)
        {
            PickleStateDict(File.OpenWrite(file), source);
        }

        /// <summary>
        /// Pickle the state_dict to a python compatible file to be loaded using `torch.load`
        /// </summary>
        /// <param name="stream">Stream of the file to write</param>
        /// <param name="source">The state_dict to pickle</param>
        public static void PickleStateDict(Stream stream, Dictionary<string, torch.Tensor> source)
        {
            // Create a new archive
            using var archive = new ZipArchive(stream, ZipArchiveMode.Create);
            // Start with writing out the pytorch version, #3
            using (var versionStream = new StreamWriter(archive.CreateEntry("model/version").Open()))
                versionStream.WriteLine(MinProducedFileFormatVersion);

            // Create our unpickler with the archive, so it can pull all the relevant files
            // using the persistentId
            var pickler = new CustomPickler(archive);

            // Wrap the source tensors in a wrapper so that we can identify them during the deconstruction
            var wrappedSource = source.ToDictionary(kvp => kvp.Key, kvp => new TensorWrapper(kvp.Value));

            // Create and dump our main data.pkl file
            using var ms = new MemoryStream();
            pickler.dump(wrappedSource, ms);

            // Copy it into the entry
            var dataPkl = archive.CreateEntry("model/data.pkl");
            using var dataStream = dataPkl.Open();
            ms.Seek(0, SeekOrigin.Begin);
            ms.CopyTo(dataStream);
        }

        /// <summary>
        /// This class implements custom behavior for pickling, specifically regarding persistent storage.
        /// In the PyTorch library, instead of serializing the tensors using Pickle, they replace the tensor
        /// objects in the pickle file with the metadata of the tensor, plus a link to an external file in the
        /// archive which contains all the byte data of the tensor.
        /// Therefore, our custom pickler does the same thing - whenever we pickle a tensor object, we break
        /// it down and store the byte data in a file in the archive and return the persistent id.
        /// </summary>
        class CustomPickler : Pickler
        {
            readonly ZipArchive _archive;
            int _tensorCount;

            public CustomPickler(ZipArchive archive)
            {
                _archive = archive;
                _tensorCount = 0;
            }

            protected override bool persistentId(object pid, out object newpid)
            {
                if (pid is not torch.Tensor) {
                    newpid = null;
                    return false;
                }

                var tensor = (torch.Tensor)pid;

                bool copied = false;
                if (tensor.device_type != DeviceType.CPU) {
                    tensor = tensor.to(torch.CPU);
                    copied = true;
                }

                // The persistentId function in pickler is a way of serializing an object using a different
                // stream and then pickling just a key representing it.
                // The data itself you store in another source. The `torch.load` function uses this functionality
                // and lists for the pid a tuple with the following items:
                // Tuple Item1: "storage"
                // Tuple Item2: storage_type (e.g., torch.LongTensor)
                // Tuple Item3: key (link to file with the byte data in the archive)
                // Tuple Item4: location (cpu/gpu)
                // Tuple Item5: numElements (number of elements in the tensor)

                // Start by serializing the object to a file in the archive
                var entry = _archive.CreateEntry($"model/data/{_tensorCount}");
                using (var stream = entry.Open())
                    stream.Write(tensor.bytes.ToArray(), 0, tensor.bytes.Length);

                // Collect the items for our persistentId, as above.
                newpid = new object[] {
                    "storage",
                    new Storage(GetStorageNameFromScalarType(tensor.dtype)), // storage_type
                    _tensorCount.ToString(), // key
                    "cpu", // location
                    tensor.NumberOfElements // numel
                };

                // Post-cleanup items
                _tensorCount++;
                if (copied) tensor.Dispose();

                return true;
            }

            static string GetStorageNameFromScalarType(torch.ScalarType storage)
            {
                return storage switch {
                    torch.ScalarType.Float64 => "DoubleStorage",
                    torch.ScalarType.Float32 => "FloatStorage",
                    torch.ScalarType.Float16 => "HalfStorage",
                    torch.ScalarType.Int64 => "LongStorage",
                    torch.ScalarType.Int32 => "IntStorage",
                    torch.ScalarType.Int16 => "ShortStorage",
                    torch.ScalarType.Int8 => "CharStorage",
                    torch.ScalarType.Byte => "ByteStorage",
                    torch.ScalarType.Bool => "BoolStorage",
                    torch.ScalarType.BFloat16 => "BFloat16Storage",
                    torch.ScalarType.ComplexFloat64 => "ComplexDoubleStorage",
                    torch.ScalarType.ComplexFloat32 => "ComplexFloatStorage",
                    _ => throw new NotImplementedException()
                };
            }
        }

        #region CustomPickleClasses
        // This region contains private custom classes which are only used for the pickling process
        // These classes are used so that we can register custom picklers/class deconstructors to them 
        // specifically. 

        /// <summary>
        /// A class wrapper for a StorageType (e.g., FloatStorage, LongStorage) so that the pickler can 
        /// assign the the storage object to our custom pickler, and we can serialize it as an python class 
        /// without any arguments.
        /// </summary>
        class Storage
        {
            public string Type { get; set; }

            public Storage(string type)
            {
                Type = type;
            }
        }

        /// <summary>
        /// A class representing an empty OrderedDict, which is used in the PyTorch serializing, for the
        /// backward_hooks reconstructions. They recommend against serializing them, and we don't support
        /// them.
        /// </summary>
        class EmptyOrderedDict
        {
            public static EmptyOrderedDict Instance => new EmptyOrderedDict();
        }

        /// <summary>
        /// A wrapper class which just contains the tensor, in order for the pickler to be able to assign
        /// the class to the TensorWrapper deconstructor
        /// </summary>
        class TensorWrapper
        {
            public torch.Tensor Tensor { get; set; }
            public TensorWrapper(torch.Tensor tensor)
            {
                Tensor = tensor;
            }
        }

        #endregion

        #region CustomPicklersAndDeconstructors

        // This region contains the custom picklers and class deconstructors.
        // The way the pickle module serializes classes is by defining the name of the module, followed
        // by the argument list for reconstructing the module. Since we use different classes in C#,
        // we defined these deconstructor to recreate the structure of the Python classes/functions being used.

        /// <summary>
        /// A custom class pickler for pickling the Storage object the way PyTorch expects to recieve, as a 
        /// class Module with no arguments for construction.
        /// The reason we used a custom pickler instead of a class deconstructor, is because the name of the
        /// module is dependent on the object being deconstructed, and using a custom pickler is more
        /// efficient than defining a class deconstructor for each storage type. 
        /// </summary>
        class StoragePickler : IObjectPickler
        {
            public void pickle(object o, Stream outs, Pickler currentPickler)
            {
                outs.WriteByte(Opcodes.GLOBAL);
                var nameBytes = Encoding.ASCII.GetBytes($"torch\n{((Storage)o).Type}\n");
                outs.Write(nameBytes, 0, nameBytes.Length);
            }
        }

        /// <summary>
        /// A class deconstructor which given an EmptyOrderedDict, it deconstructs the object into the individual
        /// members needed for reconstructing. The module in python is `collections.OrderedDict`, 
        /// And the arguments for the constructor are what are returned by the `deconstruct` function.
        /// </summary>
        class EmptyOrderedDictDeconstructor : IObjectDeconstructor
        {
            public string get_module()
            {
                return "collections";
            }

            public string get_name()
            {
                return "OrderedDict";
            }

            public object[] deconstruct(object obj)
            {
                // Empty dictionary, so the argument will be an empty array. 
                return new[] { Array.Empty<object>() };
            }
        }

        
        /// <summary>
        /// A class deconstructor which given a Tensor, it deconstructs the Tensor into the individual
        /// members needed for reconstructing. The PyTorch reconstructor is a function `torch._utils._rebuild_tensor_v2`
        /// And the arguments for that function are what are returned by the `deconstruct` function.
        /// </summary>
        class TensorWrapperDeconstructor : IObjectDeconstructor
        {
            public string get_module()
            {
                return "torch._utils";
            }

            public string get_name()
            {
                return "_rebuild_tensor_v2";
            }

            public object[] deconstruct(object obj)
            {
                var tensor = ((TensorWrapper)obj).Tensor;
                // Arg 0: Tensor
                // Arg 1: storage_offset
                // Arg 2: tensor_size (the dimension, is important)
                // Arg 3: stride (we aren't reconstructing from stride, just inserting the bytes)
                // Arg 4: requires_grad
                // Arg 5: backward_hooks, we don't support adding them in and it's not recommended in PyTorch to serialize them.
                return new object[] {
                    tensor,
                    tensor.storage_offset(),
                    tensor.shape.Select(i => (object)i).ToArray(), // cast to object so it's stored as tuple not array
                    tensor.stride().Select(i => (object)i).ToArray(), // cast to object so it's stored as tuple not array
                    tensor.requires_grad,
                    EmptyOrderedDict.Instance
                };
            }
        }

        #endregion
    }
}
