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
            /// AlexNet
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
            /// model = models.alexnet(pretrained=True)
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
            public static Modules.AlexNet alexnet(int num_classes = 1000, float dropout = 0.5f, string weights_file = null, bool skipfc = true, Device device = null)
            {
                return new Modules.AlexNet(num_classes, dropout, weights_file, skipfc, device);
            }
        }
    }

    namespace Modules
    {
        // The code here is based on
        // https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
        // Licence and copypright notice at: https://github.com/pytorch/vision/blob/main/LICENSE

        public class AlexNet : Module<Tensor, Tensor>
        {
            private readonly Module<Tensor, Tensor> features;
            private readonly Module<Tensor, Tensor> avgpool;
            private readonly Module<Tensor, Tensor> classifier;

            public AlexNet(int numClasses, float dropout = 0.5f, string weights_file = null, bool skipfc = true, Device device = null) : base(nameof(AlexNet))
            {
                features = Sequential(
                    Conv2d(3, 64, kernelSize: 11, stride: 4, padding: 2),
                    ReLU(inplace: true),
                    MaxPool2d(kernelSize: 3, stride: 2),
                    Conv2d(64, 192, kernelSize: 5, padding: 2),
                    ReLU(inplace: true),
                    MaxPool2d(kernelSize: 3, stride: 2),
                    Conv2d(192, 384, kernelSize: 3, padding: 1),
                    ReLU(inplace: true),
                    Conv2d(384, 256, kernelSize: 3, padding: 1),
                    ReLU(inplace: true),
                    Conv2d(256, 256, kernelSize: 3, padding: 1),
                    ReLU(inplace: true),
                    MaxPool2d(kernelSize: 3, stride: 2)
                );

                avgpool = AdaptiveAvgPool2d(new long[] { 6, 6 });

                classifier = Sequential(
                    Dropout(p: dropout),
                    Linear(256 * 6 * 6, 4096),
                    ReLU(inplace: true),
                    Dropout(p: dropout),
                    Linear(4096, 4096),
                    ReLU(inplace: true),
                    Linear(4096, numClasses)
                );

                RegisterComponents();

                if (!string.IsNullOrEmpty(weights_file)) {

                    this.load(weights_file, skip: skipfc ? new[] { "classifier.6.weight", "classifier.6.bias" } : null);
                }

                if (device != null && device.type == DeviceType.CUDA)
                    this.to(device);
            }

            public override Tensor forward(Tensor input)
            {
                using (var _ = NewDisposeScope()) {
                    var f = features.call(input);
                    var avg = avgpool.call(f);
                    var x = avg.flatten(1);
                    return classifier.call(x).MoveToOuterDisposeScope();
                }
            }
        }
    }
}
