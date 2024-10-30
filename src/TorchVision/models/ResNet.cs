// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

#nullable enable
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
            /// <param name="zero_init_residual">Whether to zero-initalize the residual block's norm layers.</param>
            /// <param name="groups">The number of groups.</param>
            /// <param name="width_per_group">The width of each group.</param>
            /// <param name="replace_stride_with_dilation">Each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution instead</param>
            /// <param name="norm_layer">The normalization layer to use -- a function creating a layer.</param>
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
            /// model = models.resnet18(pretrained=True)
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
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.ResNet resnet18(
                int num_classes = 1000,
                bool zero_init_residual = false,
                int groups = 1,
                int width_per_group = 64,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return Modules.ResNet.ResNet18(num_classes,
                    zero_init_residual,
                    groups,
                    width_per_group,
                    replace_stride_with_dilation,
                    norm_layer,
                    weights_file, skipfc, device);
            }

            /// <summary>
            /// ResNet-34
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="zero_init_residual">Whether to zero-initalize the residual block's norm layers.</param>
            /// <param name="groups">The number of groups.</param>
            /// <param name="width_per_group">The width of each group.</param>
            /// <param name="replace_stride_with_dilation">Each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution instead</param>
            /// <param name="norm_layer">The normalization layer to use -- a function creating a layer.</param>
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
            /// model = models.resnet34(pretrained=True)
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
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.ResNet resnet34(
                int num_classes = 1000,
                bool zero_init_residual = false,
                int groups = 1,
                int width_per_group = 64,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return Modules.ResNet.ResNet34(num_classes,
                    zero_init_residual,
                    groups,
                    width_per_group,
                    replace_stride_with_dilation,
                    norm_layer,
                    weights_file, skipfc, device);
            }

            /// <summary>
            /// ResNet-50
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="zero_init_residual">Whether to zero-initalize the residual block's norm layers.</param>
            /// <param name="groups">The number of groups.</param>
            /// <param name="width_per_group">The width of each group.</param>
            /// <param name="replace_stride_with_dilation">Each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution instead</param>
            /// <param name="norm_layer">The normalization layer to use -- a function creating a layer.</param>
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
            /// model = models.resnet50(pretrained=True)
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
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.ResNet resnet50(
                int num_classes = 1000,
                bool zero_init_residual = false,
                int groups = 1,
                int width_per_group = 64,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return Modules.ResNet.ResNet50(num_classes,
                    zero_init_residual,
                    groups,
                    width_per_group,
                    replace_stride_with_dilation,
                    norm_layer,
                    weights_file, skipfc, device);
            }

            /// <summary>
            /// Wide ResNet-50-2 model from 'Wide Residual Networks' https://arxiv.org/abs/1605.07146_
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="zero_init_residual">Whether to zero-initalize the residual block's norm layers.</param>
            /// <param name="groups">The number of groups.</param>
            /// <param name="replace_stride_with_dilation">Each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution instead</param>
            /// <param name="norm_layer">The normalization layer to use -- a function creating a layer.</param>
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
            /// model = models.wide_resnet50_2(pretrained=True)
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
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.ResNet wide_resnet50_2(
                int num_classes = 1000,
                bool zero_init_residual = false,
                int groups = 1,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return Modules.ResNet.ResNet50(num_classes,
                    zero_init_residual,
                    groups,
                    64 * 2,
                    replace_stride_with_dilation,
                    norm_layer,
                    weights_file, skipfc, device);
            }

            /// <summary>
            /// ResNeXt-50 32x4d model from
            /// `Aggregated Residual Transformation for Deep Neural Networks https://arxiv.org/abs/1611.05431
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="zero_init_residual">Whether to zero-initalize the residual block's norm layers.</param>
            /// <param name="replace_stride_with_dilation">Each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution instead</param>
            /// <param name="norm_layer">The normalization layer to use -- a function creating a layer.</param>
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
            /// model = models.resnext50_32x4d(pretrained=True)
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
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.ResNet resnext50_32x4d(
                int num_classes = 1000,
                bool zero_init_residual = false,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return Modules.ResNet.ResNet50(num_classes,
                    zero_init_residual,
                    32,
                    4,
                    replace_stride_with_dilation,
                    norm_layer,
                    weights_file, skipfc, device);
            }

            /// <summary>
            /// ResNet-101
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="zero_init_residual">Whether to zero-initalize the residual block's norm layers.</param>
            /// <param name="groups">The number of groups.</param>
            /// <param name="width_per_group">The width of each group.</param>
            /// <param name="replace_stride_with_dilation">Each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution instead</param>
            /// <param name="norm_layer">The normalization layer to use -- a function creating a layer.</param>
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
            /// model = models.resnet101(pretrained=True)
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
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.ResNet resnet101(
                int num_classes = 1000,
                bool zero_init_residual = false,
                int groups = 1,
                int width_per_group = 64,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return Modules.ResNet.ResNet101(num_classes,
                    zero_init_residual,
                    groups,
                    width_per_group,
                    replace_stride_with_dilation,
                    norm_layer,
                    weights_file, skipfc, device);
            }

            /// <summary>
            /// ResNeXt-101 32x8d model from
            /// `Aggregated Residual Transformation for Deep Neural Networks https://arxiv.org/abs/1611.05431
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="zero_init_residual">Whether to zero-initalize the residual block's norm layers.</param>
            /// <param name="replace_stride_with_dilation">Each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution instead</param>
            /// <param name="norm_layer">The normalization layer to use -- a function creating a layer.</param>
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
            /// model = models.resnext101_32x8d(pretrained=True)
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
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.ResNet resnext101_32x8d(
                int num_classes = 1000,
                bool zero_init_residual = false,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return Modules.ResNet.ResNet50(num_classes,
                    zero_init_residual,
                    32,
                    8,
                    replace_stride_with_dilation,
                    norm_layer,
                    weights_file, skipfc, device);
            }

            /// <summary>
            /// ResNeXt-101 64x4d model from
            /// `Aggregated Residual Transformation for Deep Neural Networks https://arxiv.org/abs/1611.05431
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="zero_init_residual">Whether to zero-initalize the residual block's norm layers.</param>
            /// <param name="replace_stride_with_dilation">Each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution instead</param>
            /// <param name="norm_layer">The normalization layer to use -- a function creating a layer.</param>
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
            /// model = models.resnext101_32x8d(pretrained=True)
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
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.ResNet resnext101_64x4d(
                int num_classes = 1000,
                bool zero_init_residual = false,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return Modules.ResNet.ResNet50(num_classes,
                    zero_init_residual,
                    64,
                    4,
                    replace_stride_with_dilation,
                    norm_layer,
                    weights_file, skipfc, device);
            }

            /// <summary>
            /// Wide ResNet-101-2 model from 'Wide Residual Networks' https://arxiv.org/abs/1605.07146
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="zero_init_residual">Whether to zero-initalize the residual block's norm layers.</param>
            /// <param name="groups">The number of groups.</param>
            /// <param name="replace_stride_with_dilation">Each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution instead</param>
            /// <param name="norm_layer">The normalization layer to use -- a function creating a layer.</param>
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
            /// model = models.resnet101(pretrained=True)
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
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.ResNet wide_resnet101_2(
                int num_classes = 1000,
                bool zero_init_residual = false,
                int groups = 1,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return Modules.ResNet.ResNet101(num_classes,
                    zero_init_residual,
                    groups,
                    64*2,
                    replace_stride_with_dilation,
                    norm_layer,
                    weights_file, skipfc, device);
            }

            /// <summary>
            /// ResNet-152
            /// </summary>
            /// <param name="num_classes">The number of output classes.</param>
            /// <param name="zero_init_residual">Whether to zero-initalize the residual block's norm layers.</param>
            /// <param name="groups">The number of groups.</param>
            /// <param name="width_per_group">The width of each group.</param>
            /// <param name="replace_stride_with_dilation">Each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution instead</param>
            /// <param name="norm_layer">The normalization layer to use -- a function creating a layer.</param>
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
            /// model = models.resnet152(pretrained=True)
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
            /// images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded
            /// in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            /// </remarks>
            public static Modules.ResNet resnet152(
                int num_classes = 1000,
                bool zero_init_residual = false,
                int groups = 1,
                int width_per_group = 64,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return Modules.ResNet.ResNet152(num_classes,
                    zero_init_residual,
                    groups,
                    width_per_group,
                    replace_stride_with_dilation,
                    norm_layer,
                    weights_file, skipfc, device);
            }
        }
    }

    namespace Modules
    {
        public class ResNet : Module<Tensor, Tensor>
        {
            // The code here is based on
            // https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
            // Licence and copypright notice at: https://github.com/pytorch/vision/blob/main/LICENSE

            private readonly Module<Tensor, Tensor> conv1;
            private readonly Module<Tensor, Tensor> bn1;
            private readonly Module<Tensor, Tensor> relu;
            private readonly Module<Tensor, Tensor> maxpool;

            private readonly Sequential layer1 = Sequential();
            private readonly Sequential layer2 = Sequential();
            private readonly Sequential layer3 = Sequential();
            private readonly Sequential layer4 = Sequential();

            private readonly Module<Tensor, Tensor> avgpool;
            private readonly Module<Tensor, Tensor> flatten;
            private readonly Module<Tensor, Tensor> fc;

            private readonly Func<int, Module<Tensor, Tensor>> norm_layer;

            private int in_planes = 64;
            private int dilation;
            private int groups;
            private int base_width;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    conv1.Dispose();
                    bn1.Dispose();
                    relu.Dispose();
                    maxpool.Dispose();
                    avgpool.Dispose();
                    flatten.Dispose();
                    fc.Dispose();
                    layer1.Dispose(); layer2.Dispose();
                    layer3.Dispose(); layer4.Dispose();
                }
                base.Dispose(disposing);
            }

            public static ResNet ResNet18(
                int numClasses = 1000,
                bool zero_init_residual = false,
                int groups = 1,
                int width_per_group = 64,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return new ResNet(
                    "ResNet18",
                    (in_planes, planes, stride, downsample, groups, base_width, dilation, norm_layer) => new BasicBlock(in_planes, planes, stride, downsample, groups, base_width, dilation, norm_layer),
                    BasicBlock.expansion, new int[] { 2, 2, 2, 2 },
                    numClasses,
                    zero_init_residual,
                    groups,
                    width_per_group,
                    replace_stride_with_dilation,
                    norm_layer,
                    weights_file,
                    skipfc,
                    device);
            }

            public static ResNet ResNet34(
                int numClasses = 1000,
                bool zero_init_residual = false,
                int groups = 1,
                int width_per_group = 64,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return new ResNet(
                    "ResNet34",
                    (in_planes, planes, stride, downsample, groups, base_width, dilation, norm_layer) => new BasicBlock(in_planes, planes, stride, downsample, groups, base_width, dilation, norm_layer),
                    BasicBlock.expansion, new int[] { 3, 4, 6, 3 },
                    numClasses,
                    zero_init_residual,
                    groups,
                    width_per_group,
                    replace_stride_with_dilation,
                    norm_layer,
                    weights_file,
                    skipfc,
                    device);
            }

            public static ResNet ResNet50(
                int numClasses = 1000,
                bool zero_init_residual = false,
                int groups = 1,
                int width_per_group = 64,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return new ResNet(
                    "ResNet50",
                    (in_planes, planes, stride, downsample, groups, base_width, dilation, norm_layer) => new Bottleneck(in_planes, planes, stride, downsample, groups, base_width, dilation, norm_layer),
                    Bottleneck.expansion, new int[] { 3, 4, 6, 3 },
                    numClasses,
                    zero_init_residual,
                    groups,
                    width_per_group,
                    replace_stride_with_dilation,
                    norm_layer,
                    weights_file,
                    skipfc,
                    device);
            }

            public static ResNet ResNet101(
                int numClasses = 1000,
                bool zero_init_residual = false,
                int groups = 1,
                int width_per_group = 64,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return new ResNet(
                    "ResNet101",
                    (in_planes, planes, stride, downsample, groups, base_width, dilation, norm_layer) => new Bottleneck(in_planes, planes, stride, downsample, groups, base_width, dilation, norm_layer),
                    Bottleneck.expansion, new int[] { 3, 4, 23, 3 },
                    numClasses,
                    zero_init_residual,
                    groups,
                    width_per_group,
                    replace_stride_with_dilation,
                    norm_layer,
                    weights_file,
                    skipfc,
                    device);
            }

            public static ResNet ResNet152(
                int numClasses = 1000,
                bool zero_init_residual = false,
                int groups = 1,
                int width_per_group = 64,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null)
            {
                return new ResNet(
                    "ResNet152",
                    (in_planes, planes, stride, downsample, groups, base_width, dilation, norm_layer) => new Bottleneck(in_planes, planes, stride, downsample, groups, base_width, dilation, norm_layer),
                    Bottleneck.expansion, new int[] { 3, 8, 36, 3 },
                    numClasses,
                    zero_init_residual,
                    groups,
                    width_per_group,
                    replace_stride_with_dilation,
                    norm_layer,
                    weights_file,
                    skipfc,
                    device);
            }

            public delegate Module<Tensor, Tensor> BlockFunc(
                int inplanes,
                int planes,
                int stride = 1,
                Module<Tensor, Tensor>? downsample = null,
                int groups = 1,
                int base_width = 64,
                int dilation = 1,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null);

            public ResNet(string name,
                BlockFunc block,
                int expansion,
                IList<int> layers,
                int numClasses = 1000,
                bool zero_init_residual = false,
                int groups = 1,
                int width_per_group = 64,
                (bool, bool, bool)? replace_stride_with_dilation = null,
                Func<int, Module<Tensor, Tensor>>? norm_layer = null,
                string? weights_file = null,
                bool skipfc = true,
                Device? device = null) : base(name)
            {
                norm_layer = (norm_layer is not null) ? norm_layer : (planes) => BatchNorm2d(planes);
                this.norm_layer = norm_layer;

                this.in_planes = 64;
                this.dilation = 1;
                this.groups = groups;
                this.base_width = width_per_group;

                var rswd = replace_stride_with_dilation.HasValue ? replace_stride_with_dilation.Value : (false, false, false);

                conv1 = Conv2d(3, in_planes, kernel_size: 7, stride: 2, padding: 3, bias: false);
                bn1 = norm_layer(in_planes);
                relu = ReLU(inplace: true);
                maxpool = MaxPool2d(kernel_size: 3, stride: 2, padding: 1);

                MakeLayer(layer1, block, expansion, 64, layers[0], 1);
                MakeLayer(layer2, block, expansion, 128, layers[1], 2, rswd.Item1);
                MakeLayer(layer3, block, expansion, 256, layers[2], 2, rswd.Item2);
                MakeLayer(layer4, block, expansion, 512, layers[3], 2, rswd.Item3);

                avgpool = nn.AdaptiveAvgPool2d(new long[] { 1, 1 });
                flatten = Flatten();
                fc = Linear(512 * expansion, numClasses);

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
                        case TorchSharp.Modules.GroupNorm gn:
                            torch.nn.init.constant_(gn.weight, 1);
                            torch.nn.init.constant_(gn.bias, 0);
                            break;
                        }
                    }

                    if (zero_init_residual) {
                        foreach (var (_, m) in named_modules()) {

                            switch (m) {
                            case BasicBlock bb:
                                if (bb.bn2 is TorchSharp.Modules.BatchNorm2d bb2d) {
                                    torch.nn.init.constant_(bb2d.weight, 0);
                                }
                                break;
                            case Bottleneck bn:
                                if (bn.bn3 is TorchSharp.Modules.BatchNorm2d bn2d) {
                                    torch.nn.init.constant_(bn2d.weight, 0);
                                }
                                break;
                            }
                        }

                    }

                } else {

                    this.load(weights_file!, skip: skipfc ? new[] { "fc.weight", "fc.bias" } : null);
                }

                if (device != null && device.type != DeviceType.CPU)
                    this.to(device);
            }

            private void MakeLayer(Sequential modules, BlockFunc block, int expansion, int planes, int blocks, int stride, bool dilate = false)
            {
                Sequential? downsample = null;
                var previous_dilation = this.dilation;

                if (dilate) {
                    this.dilation *= stride;
                    stride = 1;
                }

                if (stride != 1 || in_planes != planes * expansion) {
                    downsample = Sequential(
                        Conv2d(in_planes, planes * expansion, kernel_size: 1, stride: stride, bias: false),
                        norm_layer(planes * expansion)
                        );
                }

                modules.append(block(in_planes, planes, stride, downsample, groups, base_width, previous_dilation, norm_layer));

                this.in_planes = planes * expansion;

                for (int i = 1; i < blocks; i++) {
                    modules.append(block(in_planes, planes, 1, null, groups, base_width, dilation, norm_layer));
                }
            }

            public override Tensor forward(Tensor input)
            {
                using (var scope = NewDisposeScope()) {

                    var x = maxpool.call(relu.call(bn1.call(conv1.call(input))));

                    x = layer1.call(x);
                    x = layer2.call(x);
                    x = layer3.call(x);
                    x = layer4.call(x);

                    var res = fc.call(flatten.call(avgpool.call(x)));
                    scope.MoveToOuter(res);
                    return res;
                }
            }

            class BasicBlock : Module<Tensor, Tensor>
            {
                public BasicBlock(
                    int in_planes,
                    int planes,
                    int stride,
                    Module<Tensor, Tensor>? downsample = null,
                    int groups = 1,
                    int base_width = 64,
                    int dilation = 1,
                    Func<int, Module<Tensor, Tensor>>? norm_layer = null) : base("BasicBlock")
                {
                    if (groups != 1 || base_width != 64) throw new ArgumentException("BasicBlock only supports groups=1 and base_width=64");
                    if (dilation > 1) throw new NotImplementedException("dilation > 1 not supported in BasicBlock");

                    if (norm_layer is null) {
                        norm_layer = (planes) => BatchNorm2d(planes);
                    }

                    conv1 = Conv2d(in_planes, planes, kernel_size: 3, stride: stride, padding: 1, bias: false);
                    bn1 = norm_layer(planes);
                    relu1 = ReLU(inplace: true);
                    conv2 = Conv2d(planes, planes, kernel_size: 3, stride: 1, padding: 1, bias: false);
                    bn2 = norm_layer(planes);
                    this.downsample = downsample;

                    RegisterComponents();
                }

                public override Tensor forward(Tensor input)
                {
                    var identity = input;

                    var x = relu1.call(bn1.call(conv1.call(input)));
                    x = bn2.call(conv2.call(x));

                    if (downsample is not null) {
                        identity = downsample.call(input);
                    }

                    return x.add_(identity).relu_();
                }

                protected override void Dispose(bool disposing)
                {
                    if (disposing) {
                        conv1.Dispose();
                        bn1.Dispose();
                        conv2.Dispose();
                        bn2.Dispose();
                        relu1.Dispose();
                        downsample?.Dispose();
                    }
                    base.Dispose(disposing);
                }

                public static int expansion = 1;

                private readonly Module<Tensor, Tensor> conv1;
                private readonly Module<Tensor, Tensor> bn1;
                private readonly Module<Tensor, Tensor> conv2;
                internal readonly Module<Tensor, Tensor> bn2;
                private readonly Module<Tensor, Tensor> relu1;
                private readonly Module<Tensor, Tensor>? downsample;
            }

            class Bottleneck : Module<Tensor, Tensor>
            {
                public Bottleneck(
                    int in_planes,
                    int planes,
                    int stride,
                    Module<Tensor, Tensor>? downsample = null,
                    int groups = 1,
                    int base_width = 64,
                    int dilation = 1,
                    Func<int, Module<Tensor, Tensor>>? norm_layer = null) : base("Bottleneck")
                {
                    if (norm_layer is null) {
                        norm_layer = (planes) => BatchNorm2d(planes);
                    }

                    var width = (int)(planes * (base_width / 64.0)) * groups;

                    conv1 = Conv2d(in_planes, width, kernel_size: 1, bias: false);
                    bn1 = norm_layer(width);
                    relu1 = ReLU(inplace: true);
                    conv2 = Conv2d(width, width, kernel_size: 3, stride: stride, groups: groups, padding: dilation, dilation: dilation, bias: false);
                    bn2 = norm_layer(width);
                    relu2 = ReLU(inplace: true);
                    conv3 = Conv2d(width, expansion * planes, kernel_size: 1, bias: false);
                    bn3 = norm_layer(expansion * planes);

                    this.downsample = downsample;

                    RegisterComponents();
                }

                public override Tensor forward(Tensor input)
                {
                    var identity = input;

                    var x = relu1.call(bn1.call(conv1.call(input)));
                    x = relu2.call(bn2.call(conv2.call(x)));
                    x = bn3.call(conv3.call(x));

                    if (downsample is not null) {
                        identity = downsample.call(input);
                    }

                    return x.add_(identity).relu_();
                }

                protected override void Dispose(bool disposing)
                {
                    if (disposing) {
                        conv1.Dispose();
                        bn1.Dispose();
                        conv2.Dispose(); conv3.Dispose();
                        bn2.Dispose(); bn3.Dispose();
                        relu1.Dispose(); relu2.Dispose();
                        downsample?.Dispose();
                    }
                    base.Dispose(disposing);
                }

                public static int expansion = 4;

                private readonly Module<Tensor, Tensor> conv1;
                private readonly Module<Tensor, Tensor> bn1;
                private readonly Module<Tensor, Tensor> conv2;
                private readonly Module<Tensor, Tensor> bn2;
                private readonly Module<Tensor, Tensor> conv3;
                internal readonly Module<Tensor, Tensor> bn3;
                private readonly Module<Tensor, Tensor> relu1;
                private readonly Module<Tensor, Tensor> relu2;

                private readonly Module<Tensor, Tensor>? downsample;
            }
        }
    }
}
