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
    static class PyTorchUnpickler
    {
        static PyTorchUnpickler()
        {
            Unpickler.registerConstructor("torch._utils", "_rebuild_tensor", new TensorObjectConstructor());
            Unpickler.registerConstructor("torch._utils", "_rebuild_tensor_v2", new TensorObjectConstructor());
            Unpickler.registerConstructor("torch._utils", "_rebuild_parameter", new ParameterObjectConstructor());
            Unpickler.registerConstructor("collections", "OrderedDict", new OrderedDictObjectConstructor());
        }

        /// <summary>
        /// Unpickle the state_dict from a python file saved using `torch.save`
        /// </summary>
        /// <param name="file">Path to the file</param>
        /// <returns>The loaded state_dict</returns>
        public static Dictionary<string, torch.Tensor> UnpickleStateDict(string file)
        {
            return UnpickleStateDict(File.OpenRead(file));
        }

        /// <summary>
        /// Unpickle the state_dict from a python file saved using `torch.save`
        /// </summary>
        /// <param name="stream">Stream of the file to load</param>
        /// <returns>The loaded state_dict</returns>
        public static Dictionary<string, torch.Tensor> UnpickleStateDict(Stream stream)
        {
            // Make sure it's a zip file
            // If it's not, then it was saved using legacy torch save and we don't support it (yet, at least)
            // Check the local file signature
            byte[] signature = new byte[4];
            stream.Read(signature, 0, 4);
            if (signature[0] != 0x50 || signature[1] != 0x4b || signature[2] != 0x03 || signature[3] != 0x04)
                throw new NotImplementedException("The model being loaded was saved using the old PyTorch format and isn't supported in TorchSharp. Please re-save using the new PyTorch format.");

            // Open the archive, since we know it's a zip file
            stream.Seek(0, SeekOrigin.Begin);
            using var archive = new ZipArchive(stream);

            // Find the data.pkl file, this is our main file
            var pklEntry = archive.Entries.First(e => e.Name.EndsWith("data.pkl"));

            // Create our unpickler with the archive, so it can pull all the relevant files
            // using the persistentId
            var unpickler = new CustomUnpickler(archive);
            // The unpickle returns a hash mapping ["key"] to the tensor
            var stateHash = (Hashtable)unpickler.load(pklEntry.Open());

            var stateDict = new Dictionary<string, torch.Tensor>();
            foreach (string key in stateHash.Keys)
                stateDict.Add(key, (torch.Tensor)stateHash[key]);

            return stateDict;
        }

        class CustomUnpickler : Unpickler
        {
            readonly ZipArchive _archive;

            public CustomUnpickler(ZipArchive archive)
            {
                _archive = archive;
            }

            protected override object persistentLoad(object pid)
            {
                // The persistentLoad function in pickler is a way of pickling a key and then loading
                // the data yourself from another source. The `torch.save` function uses this functionality
                // and lists for the pid a tuple with the following items:
                var opid = (object[])pid;
                // tuple[0] = "storage"
                // tuple[1] = storage_type (e.g., torch.LongTensor), which is broken into module=torch, name=LongTensor
                string type = ((ClassDictConstructor)opid[1]).name;
                // key (filename in the archive)
                string archiveKey = (string)opid[2];
                // numElems (the number of elements in the tensor)
                int numElem = (int)opid[4];

                // Retrieve the entry from the archive
                var entry = _archive.Entries.First(f => f.FullName.EndsWith($"data/{archiveKey}"));

                // Convert the storage name into the relevant scalar type (e.g., LongStorage => torch.long)
                // and then check how many bytes each element is
                var dtype = GetScalarTypeFromStorageName(type);
                var elemSize = (int)torch.empty(0, dtype).ElementSize;

                int totalSize = numElem * elemSize;
                //
                // TODO: Fix this so that you can read large tensors. Right now, they are limited to 2GB
                //
                if (totalSize > int.MaxValue)
                    throw new NotImplementedException("Loading tensors larger than 2GB");

                // Read in the relevant bytes from the entry
                var bytesBuffer = new byte[totalSize];
                entry!.Open().Read(bytesBuffer, 0, totalSize);

                // Send this back, so our TensorObjectConstructor can create our torch.tensor from the object.
                return new TensorObject() {
                    data = bytesBuffer,
                    dtype = dtype
                };
            }

            static torch.ScalarType GetScalarTypeFromStorageName(string storage)
            {
                return storage switch {
                    "DoubleStorage" => torch.float64,
                    "FloatStorage" => torch.@float,
                    "HalfStorage" => torch.half,
                    "LongStorage" => torch.@long,
                    "IntStorage" => torch.@int,
                    "ShortStorage" => torch.int16,
                    "CharStorage" => torch.int8,
                    "ByteStorage" => torch.uint8,
                    "BoolStorage" => torch.@bool,
                    "BFloat16Storage" => torch.bfloat16,
                    "ComplexDoubleStorage" => torch.cdouble,
                    "ComplexFloatStorage" => torch.cfloat,
                    _ => throw new NotImplementedException()
                };
            }
        }
        class OrderedDictObjectConstructor : IObjectConstructor
        {
            public object construct(object[] args)
            {
                return new OrderedDict();
            }
        }
        class OrderedDict : Hashtable
        {
            public void __setstate__(Hashtable arg)
            {
                foreach (string key in arg.Keys) {
                    if (arg[key] is torch.Tensor)
                        this[key] = arg[key];
                }
            }
        }
        class TensorObjectConstructor : IObjectConstructor
        {
            public object construct(object[] args)
            {
                // Arg0: (byte[] data, ScalarType dtype) // returned from our custom pickler
                var arg0 = (TensorObject)args[0];
                // Arg 1: storage_offset (not relevant here)
                // Arg 2: tensor_size (the dimension, is important)
                // Arg 3: stride (we aren't reconstructing from stride, just inserting the bytes)
                // Arg 4: requires_grad
                // Arg 5: backward_hooks, we don't support adding them in and it's not recommended in PyTorch to serialize them.
                torch.Tensor t = torch.zeros(((object[])args[2]).Select(i => (long)(int)i).ToArray(), arg0.dtype, requires_grad: (bool)args[4]);
                t.bytes = arg0.data;
                return t;
            }
        }
        class ParameterObjectConstructor : IObjectConstructor
        {
            public object construct(object[] args)
            {
                // Arg0 = Tensor
                // Arg1 = requires_grad
                // If the user got here, that means that he saved the entire model and not the state dictionary
                // And we only support loading the state dict
                throw new NotImplementedException("The file trying to be load contains the entire model and not just the state_dict. Please resave use `torch.save(model.state_dict(), ...)`");
            }
        }
        class TensorObject
        {
            public byte[] data { get; set; }
            public torch.ScalarType dtype { get; set; }
        }
    }
}
