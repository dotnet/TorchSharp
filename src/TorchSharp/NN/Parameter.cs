using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public struct Parameter
    {
        public string Name { get; set; }
        public TorchTensor Tensor { get; set; }
        public bool WithGrad { get; set; }

        public Parameter(string name, TorchTensor parameter, bool? withGrad = null)
        {
            Name = name;
            Tensor = parameter;
            WithGrad = withGrad ?? parameter.IsGradRequired;
        }
    };
}
