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
            /// <param name="dropout"></param>
            /// <param name="transform_input"></param>
            /// <param name="weights_file">The location of a file containing pre-trained weights for the model.</param>
            /// <param name="skipfc">If true, the last linear layer of the classifier will not be loaded from the weights file.</param>
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
            public static Modules.InceptionV3 inception_v3(
                    int num_classes = 1000,
                    float dropout = 0.5f,
                    bool transform_input = false,
                    string weights_file = null,
                    bool skipfc = true,
                    Device device = null)
            {
                return new Modules.InceptionV3(num_classes, dropout, transform_input, weights_file, skipfc, device);
            }
        }
    }

    namespace Modules
    {
        public class InceptionV3 : Module<Tensor, Tensor>
        {
            // The code here is is loosely based on
            // https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py
            // Licence and copypright notice at: https://github.com/pytorch/vision/blob/main/LICENSE

            private readonly Module<Tensor, Tensor> Conv2d_1a_3x3;
            private readonly Module<Tensor, Tensor> Conv2d_2a_3x3;
            private readonly Module<Tensor, Tensor> Conv2d_2b_3x3;
            private readonly Module<Tensor, Tensor> maxpool1;
            private readonly Module<Tensor, Tensor> Conv2d_3b_1x1;
            private readonly Module<Tensor, Tensor> Conv2d_4a_3x3;
            private readonly Module<Tensor, Tensor> maxpool2;

            private readonly Module<Tensor, Tensor> Mixed_5b;
            private readonly Module<Tensor, Tensor> Mixed_5c;
            private readonly Module<Tensor, Tensor> Mixed_5d;
            private readonly Module<Tensor, Tensor> Mixed_6a;
            private readonly Module<Tensor, Tensor> Mixed_6b;
            private readonly Module<Tensor, Tensor> Mixed_6c;
            private readonly Module<Tensor, Tensor> Mixed_6d;
            private readonly Module<Tensor, Tensor> Mixed_6e;
            private readonly Module<Tensor, Tensor> AuxLogits;
            private readonly Module<Tensor, Tensor> Mixed_7a;
            private readonly Module<Tensor, Tensor> Mixed_7b;
            private readonly Module<Tensor, Tensor> Mixed_7c;
            private readonly AdaptiveAvgPool2d avgpool;
            private Dropout dropout;
            private readonly Linear fc;

            bool transform_input = false;

            public InceptionV3(int numClasses = 1000,
                float dropout = 0.5f,
                bool transform_input = false,
                string weights_file = null,
                bool skipfc = true,
                Device device = null) : base(nameof(InceptionV3))
            {
                this.transform_input = transform_input;

                Conv2d_1a_3x3 =  conv_block(3, 32, kernel_size: 3, stride: 2);
                Conv2d_2a_3x3 = conv_block(32, 32, kernel_size: 3);
                Conv2d_2b_3x3 = conv_block(32, 64, kernel_size: 3, padding: 1);
                maxpool1 = MaxPool2d(kernelSize: 3, stride: 2);
                Conv2d_3b_1x1 = conv_block(64, 80, kernel_size: 1);
                Conv2d_4a_3x3 = conv_block(80, 192, kernel_size: 3);
                maxpool2 = MaxPool2d(kernelSize: 3, stride: 2);

                Mixed_5b = inception_a(192, pool_features: 32);
                Mixed_5c = inception_a(256, pool_features: 64);
                Mixed_5d = inception_a(288, pool_features: 64);
                Mixed_6a = inception_b(288);
                Mixed_6b = inception_c(768, channels_7x7: 128);
                Mixed_6c = inception_c(768, channels_7x7: 160);
                Mixed_6d = inception_c(768, channels_7x7: 160);
                Mixed_6e = inception_c(768, channels_7x7: 192);
                AuxLogits = inception_aux(768, numClasses);
                Mixed_7a = inception_d(768);
                Mixed_7b = inception_e(1280);
                Mixed_7c = inception_e(2048);
                avgpool = nn.AdaptiveAvgPool2d((1, 1));
                this.dropout = nn.Dropout(p: dropout);
                fc = nn.Linear(2048, numClasses);

                RegisterComponents();

                if (string.IsNullOrEmpty(weights_file)) {

                    foreach (var (_, m) in named_modules()) {
                        switch (m) {
                        case TorchSharp.Modules.Conv2d conv:
                            torch.nn.init.kaiming_normal_(conv.weight, mode: init.FanInOut.FanOut, nonlinearity: init.NonlinearityType.ReLU);
                            break;
                        case TorchSharp.Modules.Linear ln:
                            torch.nn.init.normal_(ln.weight, 0, 0.01);
                            torch.nn.init.constant_(ln.bias, 0);
                            break;
                        case TorchSharp.Modules.BatchNorm2d bn:
                            torch.nn.init.constant_(bn.weight, 1);
                            torch.nn.init.constant_(bn.bias, 0);
                            break;
                        }
                    }
                }
                else {

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

            private Module<Tensor, Tensor> inception_a(int in_channels, int pool_features) => new InceptionA(in_channels, pool_features);
            private Module<Tensor, Tensor> inception_b(int in_channels) => new InceptionB(in_channels);
            private Module<Tensor, Tensor> inception_c(int in_channels, int channels_7x7) => new InceptionC(in_channels, channels_7x7);
            private Module<Tensor, Tensor> inception_d(int in_channels) => new InceptionD(in_channels);
            private Module<Tensor, Tensor> inception_e(int in_channels) => new InceptionE(in_channels);
            private Module<Tensor, Tensor> inception_aux(int in_channels, int num_classes) => new InceptionAux(in_channels, num_classes);

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

                    // N x 3 x 299 x 299
                    x = Conv2d_1a_3x3.call(x);
                    // N x 32 x 149 x 149
                    x = Conv2d_2a_3x3.call(x);
                    // N x 32 x 147 x 147
                    x = Conv2d_2b_3x3.call(x);
                    // N x 64 x 147 x 147
                    x = maxpool1.call(x);
                    // N x 64 x 73 x 73
                    x = Conv2d_3b_1x1.call(x);
                    // N x 80 x 73 x 73
                    x = Conv2d_4a_3x3.call(x);
                    // N x 192 x 71 x 71
                    x = maxpool2.call(x);
                    // N x 192 x 35 x 35
                    x = Mixed_5b.call(x);
                    // N x 256 x 35 x 35
                    x = Mixed_5c.call(x);
                    // N x 288 x 35 x 35
                    x = Mixed_5d.call(x);
                    // N x 288 x 35 x 35
                    x = Mixed_6a.call(x);
                    // N x 768 x 17 x 17
                    x = Mixed_6b.call(x);
                    // N x 768 x 17 x 17
                    x = Mixed_6c.call(x);
                    // N x 768 x 17 x 17
                    x = Mixed_6d.call(x);
                    // N x 768 x 17 x 17
                    x = Mixed_6e.call(x);
                    // N x 768 x 17 x 17
                    x = Mixed_7a.call(x);
                    // N x 1280 x 8 x 8
                    x = Mixed_7b.call(x);
                    // N x 2048 x 8 x 8
                    x = Mixed_7c.call(x);
                    // N x 2048 x 8 x 8
                    // Adaptive average pooling
                    x = avgpool.call(x);
                    // N x 2048 x 1 x 1
                    x = dropout.call(x);
                    // N x 2048 x 1 x 1
                    x = torch.flatten(x, 1);
                    // N x 2048
                    x = fc.call(x);
                    // N x num_classes

                    return x.MoveToOuterDisposeScope();
                }
            }

            class InceptionA : Module<Tensor, Tensor>
            {
                public InceptionA(int in_channels, int pool_features) : base("InceptionA")
                {
                    branch1x1 = conv_block(in_channels, 64, kernel_size: 1);
                    branch5x5_1 = conv_block(in_channels, 48, kernel_size: 1);
                    branch5x5_2 = conv_block(48, 64, kernel_size:5, padding: 2);
                    branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size:1);
                    branch3x3dbl_2 = conv_block(64, 96, kernel_size:3, padding: 1);
                    branch3x3dbl_3 = conv_block(96, 96, kernel_size:3, padding: 1);
                    branch_pool = conv_block(in_channels, pool_features, kernel_size:1);

                    RegisterComponents();
                }

                public override Tensor forward(Tensor x)
                {
                    var branch1x1_ = branch1x1.call(x);

                    var branch5x5 = branch5x5_1.call(x);
                    branch5x5 = branch5x5_2.call(branch5x5);

                    var branch3x3dbl = branch3x3dbl_1.call(x);
                    branch3x3dbl = branch3x3dbl_2.call(branch3x3dbl);
                    branch3x3dbl = branch3x3dbl_3.call(branch3x3dbl);

                    var branch_pool_ = functional.avg_pool2d(x, kernelSize: 3, stride: 1, padding: 1);
                    branch_pool_ = branch_pool.call(branch_pool_);

                    var outputs = new [] { branch1x1_, branch5x5, branch3x3dbl, branch_pool_ };
                    return torch.cat(outputs, 1);
                }

                private readonly Module<Tensor, Tensor> branch1x1;
                private readonly Module<Tensor, Tensor> branch5x5_1;
                private readonly Module<Tensor, Tensor> branch5x5_2;
                private readonly Module<Tensor, Tensor> branch3x3dbl_1;
                private readonly Module<Tensor, Tensor> branch3x3dbl_2;
                private readonly Module<Tensor, Tensor> branch3x3dbl_3;
                private readonly Module<Tensor, Tensor> branch_pool;
            }

            class InceptionB : Module<Tensor, Tensor>
            {
                public InceptionB(int in_channels) : base("InceptionB")
                {

                    branch3x3 = conv_block(in_channels, 384, kernel_size: 3, stride: 2);
                    branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size: 1);
                    branch3x3dbl_2 = conv_block(64, 96, kernel_size: 3, padding: 1);
                    branch3x3dbl_3 = conv_block(96, 96, kernel_size: 3, stride: 2);

                    RegisterComponents();
                }

                public override Tensor forward(Tensor x)
                {

                    var branch3x3_ = branch3x3.call(x);

                    var branch3x3dbl = branch3x3dbl_1.call(x);
                    branch3x3dbl = branch3x3dbl_2.call(branch3x3dbl);
                    branch3x3dbl = branch3x3dbl_3.call(branch3x3dbl);

                    var branch_pool = functional.max_pool2d(x, kernelSize: 3, stride: 2);

                    var outputs = new[] { branch3x3_, branch3x3dbl, branch_pool };

                    return torch.cat(outputs, 1);
                }

                private readonly Module<Tensor, Tensor> branch3x3;
                private readonly Module<Tensor, Tensor> branch3x3dbl_1;
                private readonly Module<Tensor, Tensor> branch3x3dbl_2;
                private readonly Module<Tensor, Tensor> branch3x3dbl_3;
            }

            class InceptionC : Module<Tensor, Tensor>
            {
                private readonly Module<Tensor, Tensor> branch1x1;
                private readonly Module<Tensor, Tensor> branch7x7_1;
                private readonly Module<Tensor, Tensor> branch7x7_2;
                private readonly Module<Tensor, Tensor> branch7x7_3;
                private readonly Module<Tensor, Tensor> branch7x7dbl_1;
                private readonly Module<Tensor, Tensor> branch7x7dbl_2;
                private readonly Module<Tensor, Tensor> branch7x7dbl_3;
                private readonly Module<Tensor, Tensor> branch7x7dbl_4;
                private readonly Module<Tensor, Tensor> branch7x7dbl_5;
                private readonly Module<Tensor, Tensor> branch_pool;

                public InceptionC(int in_channels, int channels_7x7) : base("InceptionC")
                {
                    branch1x1 = conv_block(in_channels, 192, kernel_size: 1);

                    var c7 = channels_7x7;
                    branch7x7_1 = conv_block(in_channels, c7, kernel_size: 1);
                    branch7x7_2 = conv_block(c7, c7, kernel_size: (1, 7), padding: (0, 3));
                    branch7x7_3 = conv_block(c7, 192, kernel_size: (7, 1), padding: (3, 0));

                    branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size: 1);
                    branch7x7dbl_2 = conv_block(c7, c7, kernel_size: (7, 1), padding: (3, 0));
                    branch7x7dbl_3 = conv_block(c7, c7, kernel_size: (1, 7), padding: (0, 3));
                    branch7x7dbl_4 = conv_block(c7, c7, kernel_size: (7, 1), padding: (3, 0));
                    branch7x7dbl_5 = conv_block(c7, 192, kernel_size: (1, 7), padding: (0, 3));

                    branch_pool = conv_block(in_channels, 192, kernel_size: 1);

                    RegisterComponents();
                }

                public override Tensor forward(Tensor x)
                {

                    var branch1x1_ = branch1x1.call(x);

                    var branch7x7 = branch7x7_1.call(x);
                    branch7x7 = branch7x7_2.call(branch7x7);
                    branch7x7 = branch7x7_3.call(branch7x7);

                    var branch7x7dbl = branch7x7dbl_1.call(x);
                    branch7x7dbl = branch7x7dbl_2.call(branch7x7dbl);
                    branch7x7dbl = branch7x7dbl_3.call(branch7x7dbl);
                    branch7x7dbl = branch7x7dbl_4.call(branch7x7dbl);
                    branch7x7dbl = branch7x7dbl_5.call(branch7x7dbl);

                    var branch_pool_ = functional.avg_pool2d(x, kernelSize: 3, stride: 1, padding: 1);
                    branch_pool_ = branch_pool.call(branch_pool_);

                    var outputs = new[] { branch1x1_, branch7x7, branch7x7dbl, branch_pool_ };
                    return torch.cat(outputs, 1);
                }
            }

            class InceptionD : Module<Tensor, Tensor>
            {
                private readonly Module<Tensor, Tensor> branch3x3_1;
                private readonly Module<Tensor, Tensor> branch3x3_2;
                private readonly Module<Tensor, Tensor> branch7x7x3_1;
                private readonly Module<Tensor, Tensor> branch7x7x3_2;
                private readonly Module<Tensor, Tensor> branch7x7x3_3;
                private readonly Module<Tensor, Tensor> branch7x7x3_4;

                public InceptionD(int in_channels) : base("InceptionD")
                {
                    branch3x3_1 = conv_block(in_channels, 192, kernel_size: 1);
                    branch3x3_2 = conv_block(192, 320, kernel_size: 3, stride: 2);

                    branch7x7x3_1 = conv_block(in_channels, 192, kernel_size: 1);
                    branch7x7x3_2 = conv_block(192, 192, kernel_size: (1, 7), padding: (0, 3));
                    branch7x7x3_3 = conv_block(192, 192, kernel_size: (7, 1), padding: (3, 0));
                    branch7x7x3_4 = conv_block(192, 192, kernel_size: 3, stride: 2);

                    RegisterComponents();
                }

                public override Tensor forward(Tensor x)
                {
                    var branch3x3 = branch3x3_1.call(x);
                    branch3x3 = branch3x3_2.call(branch3x3);

                    var branch7x7x3 = branch7x7x3_1.call(x);
                    branch7x7x3 = branch7x7x3_2.call(branch7x7x3);
                    branch7x7x3 = branch7x7x3_3.call(branch7x7x3);
                    branch7x7x3 = branch7x7x3_4.call(branch7x7x3);

                    var branch_pool = functional.max_pool2d(x, kernelSize: 3, stride: 2);
                    var outputs = new[] { branch3x3, branch7x7x3, branch_pool };


                    return torch.cat(outputs, 1);
                }
            }

            class InceptionE : Module<Tensor, Tensor>
            {
                private readonly Module<Tensor, Tensor> branch1x1;
                private readonly Module<Tensor, Tensor> branch3x3_1;
                private readonly Module<Tensor, Tensor> branch3x3_2a;
                private readonly Module<Tensor, Tensor> branch3x3_2b;
                private readonly Module<Tensor, Tensor> branch3x3dbl_1;
                private readonly Module<Tensor, Tensor> branch3x3dbl_2;
                private readonly Module<Tensor, Tensor> branch3x3dbl_3a;
                private readonly Module<Tensor, Tensor> branch3x3dbl_3b;
                private readonly Module<Tensor, Tensor> branch_pool;

                public InceptionE(int in_channels) : base("InceptionE")
                {

                    branch1x1 = conv_block(in_channels, 320, kernel_size: 1);

                    branch3x3_1 = conv_block(in_channels, 384, kernel_size: 1);
                    branch3x3_2a = conv_block(384, 384, kernel_size: (1, 3), padding: (0, 1));
                    branch3x3_2b = conv_block(384, 384, kernel_size: (3, 1), padding: (1, 0));

                    branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size: 1);
                    branch3x3dbl_2 = conv_block(448, 384, kernel_size: 3, padding: 1);
                    branch3x3dbl_3a = conv_block(384, 384, kernel_size: (1, 3), padding: (0, 1));
                    branch3x3dbl_3b = conv_block(384, 384, kernel_size: (3, 1), padding: (1, 0));

                    branch_pool = conv_block(in_channels, 192, kernel_size: 1);

                    RegisterComponents();
                }

                public override Tensor forward(Tensor x)
                {

                    var branch1x1_ = branch1x1.call(x);

                    var branch3x3 = branch3x3_1.call(x);
                    branch3x3 = torch.cat(new[] { branch3x3_2a.call(branch3x3), branch3x3_2b.call(branch3x3)}, 1);

                    var branch3x3dbl = branch3x3dbl_1.call(x);
                    branch3x3dbl = branch3x3dbl_2.call(branch3x3dbl);
                    branch3x3dbl = torch.cat(new[] { branch3x3dbl_3a.call(branch3x3dbl), branch3x3dbl_3b.call(branch3x3dbl) }, 1);

                    var branch_pool_ = functional.avg_pool2d(x, kernelSize: 3, stride: 1, padding: 1);
                    branch_pool_ = branch_pool.call(branch_pool_);

                    var outputs = new[] { branch1x1_, branch3x3, branch3x3dbl, branch_pool_ };

                    return torch.cat(outputs, 1);
                }
            }

            class InceptionAux : Module<Tensor, Tensor>
            {
                private readonly Module<Tensor, Tensor> conv0;
                private readonly Module<Tensor, Tensor> conv1;
                private readonly Module<Tensor, Tensor> fc;

                public InceptionAux(int in_channels, int num_classes) : base("InceptionAux")
                {

                    conv0 = conv_block(in_channels, 128, kernel_size: 1);
                    conv1 = conv_block(128, 768, kernel_size: 5);
                    fc = nn.Linear(768, num_classes);

                    RegisterComponents();
                }

                public override Tensor forward(Tensor x)
                {
                    // N x 768 x 17 x 17
                    x = functional.avg_pool2d(x, kernelSize: 5, stride: 3);
                    // N x 768 x 5 x 5
                    x = conv0.call(x);
                    // N x 128 x 5 x 5
                    x = conv1.call(x);
                    // N x 768 x 1 x 1
                    // Adaptive average pooling
                    x = functional.adaptive_avg_pool2d(x, (1, 1));
                    // N x 768 x 1 x 1
                    x = x.flatten(1);
                    // N x 768
                    x = fc.call(x);
                    // N x 1000

                    return x;
                }
            }
        }
    }
}
