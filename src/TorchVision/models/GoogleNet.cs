// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class models
        {
            /// <summary>
            /// ResNet-18
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="transform_input"></param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last linear layer of the classifier will not be loaded from the weights file.</param>
            /// <param name="dropout">The dropout rate for most blocks.</param>
            /// <param name="dropout_aux">The dropout rate for the aux blocks.</param>
            /// <param name="device">The device to locate the model on.</param>
            /// <remarks>
            /// Pre-trained weights may be retrieved by using Pytorch and saving the model state-dict
            /// using the exportsd.py script, then loading into the .NET instance:
            ///
            /// from torchvision import models
            /// import exportsd
            /// 
            /// model = models.inception_v3(pretrained=True)
            /// f = open("model_weights.dat", "wb")
            /// exportsd.save_state_dict(model.state_dict(), f)
            /// f.close()
            ///
            /// See also: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/saveload.md
            ///
            /// In order for the weights to be loaded, the number of classes has to be the same as
            /// in the pre-trained model, which is 1000.
            ///
            /// It is also possible to skip loading the last linear layer and use it for transfer-learning
            /// with a different number of output classes. To do so, pass skipfc=true.
            ///
            /// All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
            /// images of shape (3 x H x W), where H and W are expected to be 299x299. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.GoogleNet googlenet(
                    int num_classes = 1000,
                    bool transform_input = false,
                    string weights_file = null,
                    bool skipfc = true,
                    float dropout = 0.2f,
                    float dropout_aux = 0.7f,
                    Device device = null)
            {
                return new Modules.GoogleNet(num_classes, transform_input, weights_file, skipfc, dropout, dropout_aux, device);
            }
        }
    }

    namespace Modules
    {
        public class GoogleNet : Module<Tensor, Tensor>
        {
            // The code here is based on
            // https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py
            // Licence and copypright notice at: https://github.com/pytorch/vision/blob/main/LICENSE

            private readonly Module<Tensor, Tensor> conv1;
            private readonly Module<Tensor, Tensor> maxpool1;
            private readonly Module<Tensor, Tensor> conv2;
            private readonly Module<Tensor, Tensor> conv3;
            private readonly Module<Tensor, Tensor> maxpool2;
            private readonly Module<Tensor, Tensor> inception3a;
            private readonly Module<Tensor, Tensor> inception3b;
            private readonly Module<Tensor, Tensor> maxpool3;
            private readonly Module<Tensor, Tensor> inception4a;
            private readonly Module<Tensor, Tensor> inception4b;
            private readonly Module<Tensor, Tensor> inception4c;
            private readonly Module<Tensor, Tensor> inception4d;
            private readonly Module<Tensor, Tensor> inception4e;
            private readonly Module<Tensor, Tensor> maxpool4;
            private readonly Module<Tensor, Tensor> inception5a;
            private readonly Module<Tensor, Tensor> inception5b;
            //private readonly Module aux1;
            //private readonly Module aux2;

            private readonly AdaptiveAvgPool2d avgpool;
            private Dropout dropout;
            private readonly Linear fc;

            bool transform_input = false;

            public GoogleNet(int numClasses = 1000,
                bool transform_input = false,
                string weights_file = null,
                bool skipfc = true,
                float dropout = 0.2f,
                float dropout_aux = 0.7f,
                Device device = null) : base(nameof(GoogleNet))
            {
                this.transform_input = transform_input;

                conv1 = conv_block(3, 64, kernel_size: 7, stride: 2, padding: 3);
                maxpool1 = MaxPool2d(kernelSize: 3, stride: 2, ceilMode: true);
                conv2 = conv_block(64, 64, kernel_size: 1);
                conv3 = conv_block(64, 192, kernel_size: 3, padding: 1);
                maxpool2 = MaxPool2d(kernelSize: 3, stride: 2, ceilMode: true);

                inception3a = inception_block(192, 64, 96, 128, 16, 32, 32);
                inception3b = inception_block(256, 128, 128, 192, 32, 96, 64);
                maxpool3 = nn.MaxPool2d(3, stride: 2, ceilMode: true);

                inception4a = inception_block(480, 192, 96, 208, 16, 48, 64);
                inception4b = inception_block(512, 160, 112, 224, 24, 64, 64);
                inception4c = inception_block(512, 128, 128, 256, 24, 64, 64);
                inception4d = inception_block(512, 112, 144, 288, 32, 64, 64);
                inception4e = inception_block(528, 256, 160, 320, 32, 128, 128);
                maxpool4 = nn.MaxPool2d(2, stride: 2, ceilMode: true);

                inception5a = inception_block(832, 256, 160, 320, 32, 128, 128);
                inception5b = inception_block(832, 384, 192, 384, 48, 128, 128);

                //aux1 = inception_aux_block(512, numClasses, dropout_aux);
                //aux2 = inception_aux_block(528, numClasses, dropout_aux);

                avgpool = nn.AdaptiveAvgPool2d((1, 1));
                this.dropout = nn.Dropout(p: dropout);
                fc = nn.Linear(1024, numClasses);

                RegisterComponents();

                if (string.IsNullOrEmpty(weights_file)) {

                    foreach (var (_, m) in named_modules()) {
                        switch (m) {
                        case TorchSharp.Modules.Conv2d conv:
                            torch.nn.init.kaiming_normal_(conv.weight, mode: init.FanInOut.FanOut, nonlinearity: init.NonlinearityType.ReLU);
                            break;
                        case TorchSharp.Modules.BatchNorm2d bn:
                            torch.nn.init.constant_(bn.weight, 1);
                            torch.nn.init.constant_(bn.bias, 0);
                            break;
                        }
                    }
                } else {

                    foreach (var (_, m) in named_modules()) {
                        switch (m) {
                        case TorchSharp.Modules.Linear ln:
                            torch.nn.init.normal_(ln.weight, 0, 0.01);
                            torch.nn.init.constant_(ln.bias, 0);
                            break;
                        }
                    }
                    this.load(weights_file, skip: skipfc ? new[] { "fc.weight", "fc.bias", "AuxLogits.fc.weight", "AuxLogits.fc.bias" } : null);
                }

                if (device != null && device.type == DeviceType.CUDA)
                    this.to(device);
            }


            private static Module<Tensor, Tensor> conv_block(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0)
            {
                return Sequential(
                    ("conv", Conv2d(in_channels, out_channels, bias: false, kernelSize: kernel_size, stride: stride, padding: padding)),
                    ("bn", BatchNorm2d(out_channels, eps: 0.001)),
                    ("relu", ReLU(true))
                );
            }

            private static Module<Tensor, Tensor> conv_block(int in_channels, int out_channels, (long, long) kernel_size, (long, long)? stride = null, (long, long)? padding = null)
            {
                return Sequential(
                    ("conv", Conv2d(in_channels, out_channels, bias: false, kernelSize: kernel_size, stride: stride, padding: padding)),
                    ("bn", BatchNorm2d(out_channels, eps: 0.001)),
                    ("relu", ReLU(true))
                );
            }

            private Module<Tensor, Tensor> inception_block(int in_channels, int ch1x1, int ch3x3red,  int ch3x3, int ch5x5red, int ch5x5, int pool_proj) => new Inception(in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj);
            private Module<Tensor, Tensor> inception_aux_block(int in_channels, int num_classes, float dropout) => new InceptionAux(in_channels, num_classes, dropout);

            public override Tensor forward(Tensor x)
            {
                // Transform
                using (var scope = NewDisposeScope()) {
                    if (transform_input) {

                        var x_ch0 = torch.unsqueeze(x[(null, null), 0], 1) * (0.229f / 0.5f) + (0.485f - 0.5f) / 0.5f;
                        var x_ch1 = torch.unsqueeze(x[(null, null), 1], 1) * (0.224f / 0.5f) + (0.456f - 0.5f) / 0.5f;
                        var x_ch2 = torch.unsqueeze(x[(null, null), 2], 1) * (0.225f / 0.5f) + (0.406f - 0.5f) / 0.5f;
                        x = torch.cat(new[] { x_ch0, x_ch1, x_ch2 }, 1);
                    }

                    // N x 3 x 224 x 224
                    x = conv1.call(x);
                    // N x 64 x 112 x 112
                    x = maxpool1.call(x);
                    // N x 64 x 56 x 56
                    x = conv2.call(x);
                    // N x 64 x 56 x 56
                    x = conv3.call(x);
                    // N x 192 x 56 x 56
                    x = maxpool2.call(x);

                    // N x 192 x 28 x 28
                    x = inception3a.call(x);
                    // N x 256 x 28 x 28
                    x = inception3b.call(x);
                    // N x 480 x 28 x 28
                    x = maxpool3.call(x);
                    // N x 480 x 14 x 14
                    x = inception4a.call(x);
                    // N x 512 x 14 x 14
                    //Tensor aux1;
                    //if (this.aux1 is not null)
                    //    aux1 = this.aux1.call(x);

                    x = inception4b.call(x);
                    // N x 512 x 14 x 14
                    x = inception4c.call(x);
                    // N x 512 x 14 x 14
                    x = inception4d.call(x);
                    // N x 528 x 14 x 14
                    //Tensor aux2;
                    //if (this.aux2 is not null)
                    //    aux2 = this.aux2.call(x);

                    x = inception4e.call(x);
                    // N x 832 x 14 x 14
                    x = maxpool4.call(x);
                    // N x 832 x 7 x 7
                    x = inception5a.call(x);
                    // N x 832 x 7 x 7
                    x = inception5b.call(x);
                    // N x 1024 x 7 x 7

                    x = avgpool.call(x);
                    // N x 1024 x 1 x 1
                    x = torch.flatten(x, 1);
                    // N x 1024
                    x = dropout.call(x);
                    x = fc.call(x);
                    // N x 1000 .call(num_classes);

                    return x.MoveToOuterDisposeScope();
                }
            }

            class Inception : Module<Tensor, Tensor>
            {
                public Inception(int in_channels, int ch1x1, int ch3x3red, int ch3x3, int ch5x5red, int ch5x5, int pool_proj) : base("Inception")
                {
                    branch1 = conv_block(in_channels, ch1x1, kernel_size: 1);
                    branch2 = nn.Sequential(
                        conv_block(in_channels, ch3x3red, kernel_size: 1),
                        conv_block(ch3x3red, ch3x3, kernel_size: 3, padding: 1)
                    );
                    branch3 = nn.Sequential(
                        conv_block(in_channels, ch5x5red, kernel_size: 1),
                        conv_block(ch5x5red, ch5x5, kernel_size: 3, padding: 1)
                    );
                    branch4 = nn.Sequential(
                        nn.MaxPool2d(kernelSize: 3, stride: 1, padding: 1, ceilMode: true),
                        conv_block(in_channels, pool_proj, kernel_size: 1)
                    );
                    RegisterComponents();
                }

                public override Tensor forward(Tensor x)
                {
                    using var branch1 = this.branch1.call(x);
                    using var branch2 = this.branch2.call(x);
                    using var branch3 = this.branch3.call(x);
                    using var branch4 = this.branch4.call(x);

                    var outputs = new[] { branch1, branch2, branch3, branch4 };
                    return torch.cat(outputs, 1);
                }

                private readonly Module<Tensor, Tensor> branch1;
                private readonly Module<Tensor, Tensor> branch2;
                private readonly Module<Tensor, Tensor> branch3;
                private readonly Module<Tensor, Tensor> branch4;
            }

            class InceptionAux : Module<Tensor, Tensor>
            {
                private readonly Module<Tensor, Tensor> conv;
                private readonly Module<Tensor, Tensor> fc1;
                private readonly Module<Tensor, Tensor> fc2;
                private readonly Module<Tensor, Tensor> dropout;

                public InceptionAux(int in_channels, int num_classes, float dropout = 0.7f) : base("InceptionAux")
                {
                    conv = conv_block(in_channels, 128, kernel_size: 1);
                    fc1 = nn.Linear(2048, 1024);
                    fc2 = nn.Linear(1024, num_classes);
                    this.dropout = nn.Dropout(p: dropout);

                    RegisterComponents();
                }

                public override Tensor forward(Tensor x)
                {
                    // aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
                    x = functional.adaptive_avg_pool2d(x, 4);
                    // aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
                    x = conv.call(x);
                    // N x 128 x 4 x 4
                    x = torch.flatten(x);
                    // N x 2048
                    // Adaptive average pooling
                    x = functional.relu(fc1.call(x), inplace:true);
                    // N x 1024
                    x = dropout.call(x);
                    // N x 1024
                    x = fc2.call(x);
                    // N x 1000

                    return x;
                }
            }
        }
    }
}
