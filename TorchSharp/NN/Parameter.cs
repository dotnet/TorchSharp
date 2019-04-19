using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public struct Parameter
    {
        public string Name { get; set; }
        public ITorchTensor Tensor { get; set; }
        public bool WithGrad { get; set; }

        public Parameter(string name, ITorchTensor parameter, bool withGrad)
        {
            Name = name;
            Tensor = parameter;
            WithGrad = withGrad;
        }
    };
}
