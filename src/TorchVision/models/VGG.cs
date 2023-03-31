// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System.Collections.Generic;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class models
        {
            /// <summary>
            /// VGG-11 without batch-norm layers
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.vgg11(pretrained=True)
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
            /// as the skip list when loading.
            ///
            /// All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.VGG vgg11(int num_classes = 1000, float dropout = 0.5f, string weights_file = null, bool skipfc = true, Device device = null)
            {
                return new Modules.VGG("VGG11", num_classes, false, dropout, weights_file, skipfc, device);
            }

            /// <summary>
            /// VGG-11 with batch-norm layers
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.vgg11_bn(pretrained=True)
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
            /// as the skip list when loading.
            ///
            /// All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.VGG vgg11_bn(int num_classes = 1000, float dropout = 0.5f, string weights_file = null, bool skipfc = true, Device device = null)
            {
                return new Modules.VGG("VGG11", num_classes, true, dropout, weights_file, skipfc, device);
            }

            /// <summary>
            /// VGG-13 without batch-norm layers
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.vgg13(pretrained=True)
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
            /// as the skip list when loading.
            ///
            /// All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.VGG vgg13(int num_classes = 1000, float dropout = 0.5f, string weights_file = null, bool skipfc = true, Device device = null)
            {
                return new Modules.VGG("VGG13", num_classes, false, dropout, weights_file, skipfc, device);
            }

            /// <summary>
            /// VGG-13 with batch-norm layers
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.vgg13_bn(pretrained=True)
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
            /// as the skip list when loading.
            ///
            /// All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.VGG vgg13_bn(int num_classes = 1000, float dropout = 0.5f, string weights_file = null, bool skipfc = true, Device device = null)
            {
                return new Modules.VGG("VGG13", num_classes, true, dropout, weights_file, skipfc, device);
            }

            /// <summary>
            /// VGG-16 without batch-norm layers
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.vgg16(pretrained=True)
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
            /// as the skip list when loading.
            ///
            /// All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.VGG vgg16(int num_classes = 1000, float dropout = 0.5f, string weights_file = null, bool skipfc = true, Device device = null)
            {
                return new Modules.VGG("VGG16", num_classes, false, dropout, weights_file, skipfc, device);
            }

            /// <summary>
            /// VGG-16 with batch-norm layers
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.vgg16_bn(pretrained=True)
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
            /// as the skip list when loading.
            ///
            /// All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.VGG vgg16_bn(int num_classes = 1000, float dropout = 0.5f, string weights_file = null, bool skipfc = true, Device device = null)
            {
                return new Modules.VGG("VGG16", num_classes, true, dropout, weights_file, skipfc, device);
            }

            /// <summary>
            /// VGG-19 without batch-norm layers
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.vgg19(pretrained=True)
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
            /// as the skip list when loading.
            ///
            /// All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.VGG vgg19(int num_classes = 1000, float dropout = 0.5f, string weights_file = null, bool skipfc = true, Device device = null)
            {
                return new Modules.VGG("VGG19", num_classes, false, dropout, weights_file, skipfc, device);
            }

            /// <summary>
            /// VGG-19 with batch-norm layers
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="dropout">The dropout ratio.</param>
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
            /// model = models.vgg19_bn(pretrained=True)
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
            /// as the skip list when loading.
            ///
            /// All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.VGG vgg19_bn(int num_classes = 1000, float dropout = 0.5f, string weights_file = null, bool skipfc = true, Device device = null)
            {
                return new Modules.VGG("VGG19", num_classes, true, dropout, weights_file, skipfc, device);
            }
        }
    }

    namespace Modules
    {
        public class VGG : Module<Tensor, Tensor>
        {
            // The code here is based on
            // https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
            // Licence and copypright notice at: https://github.com/pytorch/vision/blob/main/LICENSE

            private readonly Dictionary<string, long[]> _channels = new Dictionary<string, long[]>() {
                { "VGG11", new long[] { 64, 0, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0 } },
                { "VGG13", new long[] { 64, 64, 0, 128, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0 } },
                { "VGG16", new long[] { 64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0 } },
                { "VGG19", new long[] { 64, 64, 0, 128, 128, 0, 256, 256, 256, 256, 0, 512, 512, 512, 512, 0, 512, 512, 512, 512, 0 } }
            };

            private readonly Module<Tensor, Tensor> features;
            private readonly Module<Tensor, Tensor> avgpool;
            private readonly Module<Tensor, Tensor> classifier;

            public VGG(string name,
                int numClasses,
                bool batch_norm,
                float dropout = 0.5f,
                string weights_file = null,
                bool skipfc = true,
                Device device = null) : base(name)
            {
                var layers = new List<Module<Tensor, Tensor>>();

                var channels = _channels[name];

                long in_channels = 3;

                for (var i = 0; i < channels.Length; i++) {

                    if (channels[i] == 0) {
                        layers.Add(MaxPool2d(kernelSize: 2, stride: 2));
                    } else {
                        layers.Add(Conv2d(in_channels, channels[i], kernelSize: 3, padding: 1));
                        if (batch_norm) {
                            layers.Add(BatchNorm2d(channels[i]));
                        }
                        layers.Add(ReLU(inplace: true));
                        in_channels = channels[i];
                    }
                }

                features = Sequential(layers);

                avgpool = AdaptiveAvgPool2d(new[] { 7L, 7L });

                classifier = Sequential(
                    Linear(512 * 7 * 7, 4096),
                    ReLU(true),
                    Dropout(p: dropout),
                    Linear(4096, 4096),
                    ReLU(true),
                    Dropout(p: dropout),
                    Linear(4096, numClasses)
                    );

                RegisterComponents();

                if (string.IsNullOrEmpty(weights_file)) {

                    foreach (var (_, m) in named_modules()) {
                        switch (m) {
                        // This test must come before the Tensor test
                        case TorchSharp.Modules.Conv2d conv:
                            torch.nn.init.kaiming_normal_(conv.weight, mode: init.FanInOut.FanOut, nonlinearity: init.NonlinearityType.ReLU);
                            if (conv.bias is not null && !conv.bias.IsInvalid) {
                                torch.nn.init.constant_(conv.bias, 0);
                            }
                            break;
                        case TorchSharp.Modules.BatchNorm2d bn:
                            torch.nn.init.constant_(bn.weight, 1);
                            torch.nn.init.constant_(bn.bias, 0);
                            break;
                        case TorchSharp.Modules.Linear ln:
                            torch.nn.init.normal_(ln.weight, 0, 0.01);
                            torch.nn.init.constant_(ln.bias, 0);
                            break;
                        }
                    }
                } else {

                    foreach (var (_, m) in named_modules()) {
                        switch (m) {
                        // This test must come before the Tensor test
                        case TorchSharp.Modules.Linear ln:
                            torch.nn.init.normal_(ln.weight, 0, 0.01);
                            torch.nn.init.constant_(ln.bias, 0);
                            break;
                        }
                    }

                    this.load(weights_file, skip: skipfc ? new[] { "classifier.6.weight", "classifier.6.bias" } : null);
                }

                if (device != null && device.type == DeviceType.CUDA)
                    this.to(device);
            }

            public override Tensor forward(Tensor input)
            {
                using (var _ = NewDisposeScope()) {
                    input = features.call(input);
                    input = avgpool.call(input).flatten(1);
                    return classifier.call(input).MoveToOuterDisposeScope();
                }
            }
        }
    }
}
