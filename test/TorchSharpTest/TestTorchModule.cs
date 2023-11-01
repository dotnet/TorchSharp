using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Xunit;
using TorchSharp.Utils;
using TorchSharp;

namespace TorchSharpTest
{
    [Collection("Sequential")]
    public class TestTorchModule
    {
        private class CustomNameModel : nn.Module<Tensor, Tensor>
        {
            [ComponentName(Name = "custom_linear")]
            private Linear _linear1;
            private Linear _linear2;

            public CustomNameModel(string name) : base(name)
            {
                _linear1 = Linear(5, 5, hasBias: false);
                _linear2 = Linear(5, 5, hasBias: false);

                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                return _linear2.forward(_linear1.forward(input));
            }
        }

        [Fact]
        public void TestCustomComponentName()
        {
            var model = new CustomNameModel("CustomNameModel");

            var sd = model.state_dict();
            // Make sure that it's saved in the state_dict correctly, with and without custom attribute
            Assert.True(sd.ContainsKey("custom_linear.weight"));
            Assert.True(sd.ContainsKey("_linear2.weight"));
        }

        [Fact]
        public void TestCustomComponentNameAfterToEpilogue()
        {
            var model = new CustomNameModel("CustomNameModel").to(ScalarType.BFloat16);
            
            var sd = model.state_dict();
            // Make sure that it's saved in the state_dict correctly, with and without custom attribute
            Assert.True(sd.ContainsKey("custom_linear.weight"));
            Assert.True(sd.ContainsKey("_linear2.weight"));
        }
    }
}
