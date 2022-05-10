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
            using var model = resnet18();
            var sd = model.state_dict();
            Assert.Equal(122, sd.Count);
        }

        [Fact]
        public void TestResNet34()
        {
            using var model = resnet34();
            var sd = model.state_dict();
            Assert.Equal(218, sd.Count);
        }

        [Fact]
        public void TestResNet50()
        {
            using var model = resnet50();
            var sd = model.state_dict();
            Assert.Equal(320, sd.Count);
        }

        [Fact]
        public void TestResNet101()
        {
            using var model = resnet101();
            var sd = model.state_dict();
            Assert.Equal(626, sd.Count);
        }

        [Fact]
        public void TestResNet152()
        {
            using var model = resnet152();
            var sd = model.state_dict();
            Assert.Equal(932, sd.Count);
        }

        [Fact]
        public void TestAlexNet()
        {
            using var model = alexnet();
            var sd = model.state_dict();
            Assert.Equal(16, sd.Count);
        }

        [Fact]
        public void TestVGG11()
        {
            {
                using var model = vgg11();
                var sd = model.state_dict();
                Assert.Equal(22, sd.Count);
            }
            {
                using var model = vgg11_bn();
                var sd = model.state_dict();
                Assert.Equal(62, sd.Count);
            }
        }

        [Fact]
        public void TestVGG13()
        {
            {
                using var model = vgg13();
                var sd = model.state_dict();
                Assert.Equal(26, sd.Count);
            }
            {
                using var model = vgg13_bn();
                var sd = model.state_dict();
                Assert.Equal(76, sd.Count);
            }
        }

        [Fact]
        public void TestVGG16()
        {
            {
                using var model = vgg16();
                var sd = model.state_dict();
                Assert.Equal(32, sd.Count);
            }
            {
                using var model = vgg16_bn();
                var sd = model.state_dict();
                Assert.Equal(97, sd.Count);
            }
        }

        [Fact]
        public void TestVGG19()
        {
            {
                using var model = vgg19();
                var sd = model.state_dict();
                Assert.Equal(38, sd.Count);
            }
            {
                using var model = vgg19_bn();
                var sd = model.state_dict();
                Assert.Equal(118, sd.Count);
            }
        }

        [Fact]
        public void TestInception()
        {
            using var model = inception_v3();
            var sd = model.state_dict();
            Assert.Equal(580, sd.Count);
        }

        [Fact]
        public void TestGoogLeNet()
        {
            using var model = googlenet();
            var sd = model.state_dict();
            Assert.Equal(344, sd.Count);
        }
    }
}
