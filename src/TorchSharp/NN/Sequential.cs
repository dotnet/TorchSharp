
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Sequential : ProvidedModule
    {
        internal Sequential(IntPtr handle) : base(handle)
        {
        }

        public Sequential(IEnumerable<Module> modules) : base(IntPtr.Zero)
        {
            foreach (var module in modules)
            {
                RegisterModule(module);
            }
        }

        public override TorchTensor Forward(TorchTensor tensor)
        {
            if (!Modules.Any())
            {
                throw new ArgumentException("Cannot do forward pass over empty Sequence module.");
            }

            var (head, tail) = Modules;
            var result = head.Forward(tensor);

            foreach (var module in tail)
            {
                var tmp = module.Forward(result);
                result.Dispose();
                result = tmp;
            }

            return result;
        }

        public override void Save(String location)
        {
            int count = 0;
            using (StreamWriter writer = File.CreateText(location + "/model-list.txt"))
            {
                foreach (var module in Modules)
                {
                    var actualLocation = module.GetType() + "\t" + count++;
                    writer.WriteLine(actualLocation);
                    module.Save(actualLocation);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSNN_load_module(string location);

        public new static Module Load(String location)
        {
            Contract.Assert(File.Exists(location + "/model-list.txt"));
            var modules = new List<Module>();

            foreach (var line in File.ReadLines(location + "/model-list.txt"))
            {
                var splitted = line.Split('\t');

                Contract.Assert(splitted.Count() == 2);
                
                switch(splitted[0])
                {
                    case "Linear":
                        modules.Add(TorchSharp.NN.Linear.Load(line));
                        break;
                    case "Conv2d":
                        modules.Add(TorchSharp.NN.Conv2D.Load(line));
                        break;
                    default:
                        throw new ArgumentException(@"Module type {splitted[0]} not found.");
                }
            }

            return new Sequential(modules);
        }

        public override void ZeroGrad()
        {
            foreach (var module in Modules)
            {
                module.ZeroGrad();
            }
        }

        public override IEnumerable<string> GetModules()
        {
            List<string> result = new List<string>();

            foreach (var module in Modules)
            {
                result.Add(module.GetName());
            }

            return result;
        }

        public override void Train()
        {
            foreach (var module in Modules)
            {
                module.Train();
            }
        }

        public override void Eval()
        {
            foreach (var module in Modules)
            {
                module.Eval();
            }
        }
    }
}
