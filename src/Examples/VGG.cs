using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp.NN;
using TorchSharp.Tensor;

namespace Examples
{
    /// <summary>
    ///     
    /// </summary>
    class VGG : TorchSharp.NN.Module
    {
        Sequential sequential;
        private readonly static Dictionary<string, List<object>> cfgs = new Dictionary<string, List<object>>();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="VGGName"></param>
        public VGG(string VGGName) : base()
        {
            // super(VGG, self).__init__()  dont really need it, i think
            List<object> cfg = cfgs.GetValueOrDefault(VGGName);
            this.sequential = MakeSequential(cfg);

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private override TorchTensor Forward(TorchTensor input)
        {
        }

        private Sequential MakeSequential(List<object> cfg)
        {
            List<Module> layers = new List<Module>();
            int in_channels = 3;

            foreach (object x in cfg)
            {
                if (x.ToString() == "M")
                    layers.Add(Module.MaxPool2D(new long[2], new long[1]));
                else
                {
                    layers.Add(Module.Conv2D(in_channels, (long)x, 3, padding: 1));
                    //layers.Add(Module.Bat);
                    layers.Add(Module.Relu(inPlace: true));
                    in_channels = (int) x;
                }
            }
            //layers.Add(Module.AdaptiveAvgPool2D( )

            return Module.Sequential(layers.ToArray());
        }

    }
}
