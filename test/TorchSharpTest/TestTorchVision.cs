using System.Linq;
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

            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("bn1", names[1]),
                () => Assert.Equal("relu", names[2]),
                () => Assert.Equal("maxpool", names[3]),
                () => Assert.Equal("layer1", names[4]),
                () => Assert.Equal("layer2", names[5]),
                () => Assert.Equal("layer3", names[6]),
                () => Assert.Equal("layer4", names[7]),
                () => Assert.Equal("avgpool", names[8]),
                () => Assert.Equal("flatten", names[9]),
                () => Assert.Equal("fc", names[10])
            );
        }

        [Fact]
        public void TestResNet34()
        {
            using var model = resnet34();
            var sd = model.state_dict();
            Assert.Equal(218, sd.Count);

            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("bn1", names[1]),
                () => Assert.Equal("relu", names[2]),
                () => Assert.Equal("maxpool", names[3]),
                () => Assert.Equal("layer1", names[4]),
                () => Assert.Equal("layer2", names[5]),
                () => Assert.Equal("layer3", names[6]),
                () => Assert.Equal("layer4", names[7]),
                () => Assert.Equal("avgpool", names[8]),
                () => Assert.Equal("flatten", names[9]),
                () => Assert.Equal("fc", names[10])
            );
        }

        [Fact]
        public void TestResNet50()
        {
            using var model = resnet50();
            var sd = model.state_dict();
            Assert.Equal(320, sd.Count);

            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("bn1", names[1]),
                () => Assert.Equal("relu", names[2]),
                () => Assert.Equal("maxpool", names[3]),
                () => Assert.Equal("layer1", names[4]),
                () => Assert.Equal("layer2", names[5]),
                () => Assert.Equal("layer3", names[6]),
                () => Assert.Equal("layer4", names[7]),
                () => Assert.Equal("avgpool", names[8]),
                () => Assert.Equal("flatten", names[9]),
                () => Assert.Equal("fc", names[10])
            );
        }

        [Fact]
        public void TestResNet101()
        {
            using var model = resnet101();
            var sd = model.state_dict();
            Assert.Equal(626, sd.Count);

            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("bn1", names[1]),
                () => Assert.Equal("relu", names[2]),
                () => Assert.Equal("maxpool", names[3]),
                () => Assert.Equal("layer1", names[4]),
                () => Assert.Equal("layer2", names[5]),
                () => Assert.Equal("layer3", names[6]),
                () => Assert.Equal("layer4", names[7]),
                () => Assert.Equal("avgpool", names[8]),
                () => Assert.Equal("flatten", names[9]),
                () => Assert.Equal("fc", names[10])
            );
        }

        [Fact]
        public void TestResNet152()
        {
            using var model = resnet152();
            var sd = model.state_dict();
            Assert.Equal(932, sd.Count);

            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("bn1", names[1]),
                () => Assert.Equal("relu", names[2]),
                () => Assert.Equal("maxpool", names[3]),
                () => Assert.Equal("layer1", names[4]),
                () => Assert.Equal("layer2", names[5]),
                () => Assert.Equal("layer3", names[6]),
                () => Assert.Equal("layer4", names[7]),
                () => Assert.Equal("avgpool", names[8]),
                () => Assert.Equal("flatten", names[9]),
                () => Assert.Equal("fc", names[10])
            );
        }

        [Fact]
        public void TestAlexNet()
        {
            using var model = alexnet();
            var sd = model.state_dict();
            Assert.Equal(16, sd.Count);
            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("features", names[0]),
                () => Assert.Equal("avgpool", names[1]),
                () => Assert.Equal("classifier", names[2])
            );
        }

#if DEBUG
        [Fact(Skip = "The test takes too long to run -- across the various VGG versions, 2/3 of overall test time is spent here.")]
#else
        [Fact]
#endif
        public void TestVGG11()
        {
            {
                using var model = vgg11();
                var sd = model.state_dict();
                Assert.Equal(22, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
            {
                using var model = vgg11_bn();
                var sd = model.state_dict();
                Assert.Equal(62, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
        }

#if DEBUG
        [Fact(Skip = "The test takes too long to run -- across the various VGG versions, 2/3 of overall test time is spent here.")]
#else
        [Fact]
#endif
        public void TestVGG13()
        {
            {
                using var model = vgg13();
                var sd = model.state_dict();
                Assert.Equal(26, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
            {
                using var model = vgg13_bn();
                var sd = model.state_dict();
                Assert.Equal(76, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
        }

#if DEBUG
        [Fact(Skip = "The test takes too long to run -- across the various VGG versions, 2/3 of overall test time is spent here.")]
#else
        [Fact]
#endif
        public void TestVGG16()
        {
            {
                using var model = vgg16();
                var sd = model.state_dict();
                Assert.Equal(32, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
            {
                using var model = vgg16_bn();
                var sd = model.state_dict();
                Assert.Equal(97, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
        }

#if DEBUG
        [Fact(Skip = "The test takes too long to run -- across the various VGG versions, 2/3 of overall test time is spent here.")]
#else
        [Fact]
#endif
        public void TestVGG19()
        {
            {
                using var model = vgg19();
                var sd = model.state_dict();
                Assert.Equal(38, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
            {
                using var model = vgg19_bn();
                var sd = model.state_dict();
                Assert.Equal(118, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
        }

        [Fact]
        public void TestInception()
        {
            using var model = inception_v3();
            var sd = model.state_dict();
            Assert.Equal(580, sd.Count);
            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("Conv2d_1a_3x3", names[0]),
                () => Assert.Equal("Conv2d_2a_3x3", names[1]),
                () => Assert.Equal("Conv2d_2b_3x3", names[2]),
                () => Assert.Equal("maxpool1", names[3]),
                () => Assert.Equal("Conv2d_3b_1x1", names[4]),
                () => Assert.Equal("Conv2d_4a_3x3", names[5]),
                () => Assert.Equal("maxpool2", names[6]),
                () => Assert.Equal("Mixed_5b", names[7]),
                () => Assert.Equal("Mixed_5c", names[8]),
                () => Assert.Equal("Mixed_5d", names[9]),
                () => Assert.Equal("Mixed_6a", names[10]),
                () => Assert.Equal("Mixed_6b", names[11]),
                () => Assert.Equal("Mixed_6c", names[12]),
                () => Assert.Equal("Mixed_6d", names[13]),
                () => Assert.Equal("Mixed_6e", names[14]),
                () => Assert.Equal("AuxLogits", names[15]),
                () => Assert.Equal("Mixed_7a", names[16]),
                () => Assert.Equal("Mixed_7b", names[17]),
                () => Assert.Equal("Mixed_7c", names[18]),
                () => Assert.Equal("avgpool", names[19]),
                () => Assert.Equal("dropout", names[20]),
                () => Assert.Equal("fc", names[21])
            );
        }

        [Fact]
        public void TestGoogLeNet()
        {
            using var model = googlenet();
            var sd = model.state_dict();
            Assert.Equal(344, sd.Count);
            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("maxpool1", names[1]),
                () => Assert.Equal("conv2", names[2]),
                () => Assert.Equal("conv3", names[3]),
                () => Assert.Equal("maxpool2", names[4]),
                () => Assert.Equal("inception3a", names[5]),
                () => Assert.Equal("inception3b", names[6]),
                () => Assert.Equal("maxpool3", names[7]),
                () => Assert.Equal("inception4a", names[8]),
                () => Assert.Equal("inception4b", names[9]),
                () => Assert.Equal("inception4c", names[10]),
                () => Assert.Equal("inception4d", names[11]),
                () => Assert.Equal("inception4e", names[12]),
                () => Assert.Equal("maxpool4", names[13]),
                () => Assert.Equal("inception5a", names[14]),
                () => Assert.Equal("inception5b", names[15]),
                () => Assert.Equal("avgpool", names[16]),
                () => Assert.Equal("dropout", names[17]),
                () => Assert.Equal("fc", names[18])
            );
        }

        [Fact]
        public void TestMobileNetV2()
        {
            using var model = mobilenet_v2();
            var sd = model.state_dict();
            Assert.Equal(314, sd.Count);
            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("classifier", names[0]),
                () => Assert.Equal("features", names[1])
            );
        }

        [Fact]
        public void TestMobileNetV3()
        {
            using (var model = mobilenet_v3_large()) {
                var sd = model.state_dict();
                Assert.Equal(312, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("avgpool", names[0]),
                    () => Assert.Equal("classifier", names[1]),
                    () => Assert.Equal("features", names[2])
                );
            }

            using (var model = mobilenet_v3_small()) {
                var sd = model.state_dict();
                Assert.Equal(244, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("avgpool", names[0]),
                    () => Assert.Equal("classifier", names[1]),
                    () => Assert.Equal("features", names[2])
                );
            }
        }
    }
}
