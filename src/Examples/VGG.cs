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
        Module classifier;
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
            classifier = Linear(320, 50);

    }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public override TorchTensor Forward(TorchTensor input)
        {
            TorchTensor res = this.sequential.Forward(input);
            res = res.View(new long[] { -1, 0 });
            return classifier.Forward(res);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="cfg"></param>
        /// <returns></returns>
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
            // layers.Add(Module.AdaptiveAvgPool2D();

            return Module.Sequential(layers.ToArray());
        }

    }
}
