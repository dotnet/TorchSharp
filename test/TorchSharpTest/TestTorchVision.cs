using System.Collections.Generic;
using TorchSharp.Modules;
using static TorchSharp.torchvision.models;
using Xunit;


namespace TorchSharp
{
#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
    public class TestTorchVision
    { 
        [Fact]
        public void TestResNet18()
        {
            var model = resnet18();
            var sd = model.state_dict();
            Assert.Equal(122, sd.Count);
        }

        [Fact]
        public void TestResNet34()
        {
            var model = resnet34();
            var sd = model.state_dict();
            Assert.Equal(218, sd.Count);
        }

        [Fact]
        public void TestResNet50()
        {
            var model = resnet50();
            var sd = model.state_dict();
            Assert.Equal(320, sd.Count);
        }

        [Fact]
        public void TestResNet101()
        {
            var model = resnet101();
            var sd = model.state_dict();
            Assert.Equal(626, sd.Count);
        }

        [Fact]
        public void TestResNet152()
        {
            var model = resnet152();
            var sd = model.state_dict();
            Assert.Equal(932, sd.Count);
        }

        [Fact]
        public void TestAlexNet()
        {
            var model = alexnet();
            var sd = model.state_dict();
            Assert.Equal(16, sd.Count);
        }

        [Fact]
        public void TestVGG11()
        {
            {
                var model = vgg11();
                var sd = model.state_dict();
                Assert.Equal(22, sd.Count);
            }
            {
                var model = vgg11_bn();
                var sd = model.state_dict();
                Assert.Equal(62, sd.Count);
            }
        }

        [Fact]
        public void TestVGG13()
        {
            {
                var model = vgg13();
                var sd = model.state_dict();
                Assert.Equal(26, sd.Count);
            }
            {
                var model = vgg13_bn();
                var sd = model.state_dict();
                Assert.Equal(76, sd.Count);
            }
        }

        [Fact]
        public void TestVGG16()
        {
            {
                var model = vgg16();
                var sd = model.state_dict();
                Assert.Equal(32, sd.Count);
            }
            {
                var model = vgg16_bn();
                var sd = model.state_dict();
                Assert.Equal(97, sd.Count);
            }
        }

        [Fact]
        public void TestVGG19()
        {
            {
                var model = vgg19();
                var sd = model.state_dict();
                Assert.Equal(38, sd.Count);
            }
            {
                var model = vgg19_bn();
                var sd = model.state_dict();
                Assert.Equal(118, sd.Count);
            }
        }
    }
}
