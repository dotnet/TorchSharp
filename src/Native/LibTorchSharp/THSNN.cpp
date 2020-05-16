// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "THSNN.h"

#include <torch/nn/init.h>

// Wrapper class used to enable the addition of parameters.
class ModuleWrapper : torch::nn::Module
{
    public :
        ModuleWrapper(
            const char ** names, 
            at::Tensor ** parameters, 
            const bool * require_grad, const int length)
            : torch::nn::Module("Module")
        {
            for (int i = 0; i < length; i++)
            {
                register_parameter(names[i], *parameters[i], require_grad[i]);
            }
        }
};

// API

NNModule THSNN_ReLU_ctor(bool inplace)
{
	auto options = torch::nn::ReLUOptions(inplace);
	CATCH_RETURN(NNModule, new std::shared_ptr<torch::nn::Module>(torch::nn::ReLU(options).ptr()));
}

Tensor THSNN_Relu_forward(const NNModule module, const Tensor tensor)
{
	CATCH_RETURN_TENSOR((*module)->as<torch::nn::ReLU>()->forward(*tensor));
}

NNModule THSNN_Dropout_ctor(double probability)
{
	auto options = torch::nn::DropoutOptions(probability);
	CATCH_RETURN(NNModule, new std::shared_ptr<torch::nn::Module>(torch::nn::Dropout(options).ptr()));
}

Tensor THSNN_Dropout_forward(const NNModule module, const Tensor tensor)
{
	CATCH_RETURN_TENSOR((*module)->as<torch::nn::Dropout>()->forward(*tensor));
}

NNModule THSNN_FeatureAlphaDropout_ctor(double probability)
{
	auto options = torch::nn::FeatureAlphaDropoutOptions(probability);
	CATCH_RETURN(NNModule, new std::shared_ptr<torch::nn::Module>(torch::nn::FeatureAlphaDropout(options).ptr()));
}

Tensor THSNN_FeatureAlphaDropout_forward(const NNModule module, const Tensor tensor)
{
	CATCH_RETURN_TENSOR((*module)->as<torch::nn::FeatureAlphaDropout>()->forward(*tensor));
}

NNModule THSNN_LogSoftMax_ctor(int64_t dim)
{
	auto options = torch::nn::LogSoftmaxOptions(dim);
	CATCH_RETURN(NNModule, new std::shared_ptr<torch::nn::Module>(torch::nn::LogSoftmax(options).ptr()));
}

Tensor THSNN_LogSoftMax_forward(const NNModule module, const Tensor tensor)
{
	CATCH_RETURN_TENSOR((*module)->as<torch::nn::LogSoftmax>()->forward(*tensor));
}

NNModule THSNN_AvgPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength)
{
	auto options = torch::nn::AvgPool2dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
	if (stride)
		options = options.stride(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
	CATCH_RETURN(NNModule, new std::shared_ptr<torch::nn::Module>(torch::nn::AvgPool2d(options).ptr()));
}

Tensor THSNN_AvgPool2d_forward(const NNModule module, const Tensor tensor)
{
	CATCH_RETURN_TENSOR((*module)->as<torch::nn::AvgPool2d>()->forward(*tensor));
}

NNModule THSNN_AdaptiveAvgPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength)
{
	auto options = torch::nn::AdaptiveAvgPool2dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
	CATCH_RETURN(NNModule, new std::shared_ptr<torch::nn::Module>(torch::nn::AdaptiveAvgPool2d(options).ptr()));
}

Tensor THSNN_AdaptiveAvgPool2d_forward(const NNModule module, const Tensor tensor)
{
	CATCH_RETURN_TENSOR((*module)->as<torch::nn::AdaptiveAvgPool2d>()->forward(*tensor));
}

NNModule THSNN_MaxPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength)
{
	auto options = torch::nn::MaxPool2dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
	if (stride)
		options = options.stride(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
	CATCH_RETURN(NNModule, new std::shared_ptr<torch::nn::Module>(torch::nn::MaxPool2d(options).ptr()));
}

Tensor THSNN_MaxPool2d_forward(const NNModule module, const Tensor tensor)
{
	CATCH_RETURN_TENSOR((*module)->as<torch::nn::MaxPool2d>()->forward(*tensor));
}

NNModule THSNN_Linear_ctor(const int64_t input_size, const int64_t output_size, const bool bias)
{
	auto options = torch::nn::LinearOptions(input_size, output_size).bias(bias);
	CATCH_RETURN(NNModule, new std::shared_ptr<torch::nn::Module>(torch::nn::Linear(options).ptr()));
}

Tensor THSNN_Linear_forward(const NNModule module, const Tensor tensor)
{
	CATCH_RETURN_TENSOR((*module)->as<torch::nn::Linear>()->forward(*tensor));
}

Tensor THSNN_Linear_bias(const NNModule module)
{
	CATCH_RETURN_TENSOR((*module)->as<torch::nn::Linear>()->bias);
}

void THSNN_Linear_set_bias(const NNModule module, const Tensor bias)
{
	CATCH(
		(*module)->as<torch::nn::Linear>()->bias = *bias;
	)
}

Tensor THSNN_Linear_weight(const NNModule module)
{
	CATCH_RETURN_TENSOR((*module)->as<torch::nn::Linear>()->weight);
}

void THSNN_Linear_set_weight(const NNModule module, const Tensor weight)
{
	CATCH(
		(*module)->as<torch::nn::Linear>()->weight = *weight;
	)
}

NNModule THSNN_Conv2d_ctor(const int64_t inputChannel, const int64_t outputChannel, const int64_t kernelSize, const int64_t stride, const int64_t padding)
{
	auto options = torch::nn::Conv2dOptions(inputChannel, outputChannel, kernelSize).stride(stride).padding(padding);
	CATCH_RETURN(NNModule, new std::shared_ptr<torch::nn::Module>(torch::nn::Conv2d(options).ptr()));
}

Tensor THSNN_Conv2d_forward(const NNModule module, const Tensor tensor)
{
	CATCH_RETURN_TENSOR((*module)->as<torch::nn::Conv2d>()->forward(*tensor));
}

NNModule THSNN_Sequential_ctor()
{
	CATCH_RETURN(NNModule, new std::shared_ptr<torch::nn::Module>(torch::nn::Sequential().ptr()));
}

Tensor THSNN_Sequential_forward(const NNModule module, const Tensor tensor)
{
	CATCH_RETURN_TENSOR((*module)->as<torch::nn::Sequential>()->forward(*tensor));
}

NNModule THSNN_new_module(const char ** names, at::Tensor ** parameters, const bool * require_grad, const int length)
{
	NNModule res;
	CATCH(
		torch::nn::Module* module = (torch::nn::Module*)new ModuleWrapper(names, parameters, require_grad, length);
        res = new std::shared_ptr<torch::nn::Module>(module);
    );
	return res;
}

NNModule THSNN_Module_load(const char * location, const char * name)
{
	NNModule res;
	CATCH(
		auto module = new torch::nn::Module();
        auto input = torch::serialize::InputArchive();

        input.load_from(location);
        module->load(input);
        res = new std::shared_ptr<torch::nn::Module>(module);
    );
	return res;
}

int THSNN_Module_has_parameter(const NNModule module, const char * name)
{
	CATCH_RETURN(int, (*module)->named_parameters().contains(name));
}

Tensor THSNN_Module_get_parameter(const NNModule module, const char * name)
{
	CATCH_RETURN_TENSOR(*(*module)->named_parameters().find(name));
}

void THSNN_Module_get_parameters(const NNModule module, Tensor* (*allocator1)(size_t length))
{
    auto parameters = (*module)->named_parameters();
    Tensor * result1 = allocator1(parameters.size());

    for (size_t i = 0; i < parameters.size(); i++)
    {
        result1[i] = new torch::Tensor(parameters[i].value());
    }
}

void THSNN_Module_get_named_parameters(const NNModule module, Tensor* (*allocator1)(size_t length), const char** (*allocator2)(size_t length))
{
    auto parameters = (*module)->named_parameters();
    Tensor * result1 = allocator1(parameters.size());
    const char ** result2 = allocator2(parameters.size());

    for (size_t i = 0; i < parameters.size(); i++)
    {
        result1[i] = new torch::Tensor(parameters[i].value());
        result2[i] = make_sharable_string(parameters[i].key());
    }
}

int THSNN_Module_is_training(NNModule module)
{
    return (*module)->is_training();
}

void THSNN_Module_train(NNModule module)
{
    (*module)->train();
}

void THSNN_Module_eval(NNModule module)
{
    (*module)->eval();
}

long THSNN_Module_children_size(const NNModule module)
{
    return (*module)->children().size();
}

const char * THSNN_getChildModuleName(const NNModule module, const int index)
{
    return make_sharable_string((*module)->children()[index]->name());
}

const char * THSNN_Module_name(const NNModule module)
{
    return make_sharable_string((*module)->name());
}


void THSNN_Module_zeroGrad(const NNModule module)
{
    (*module)->zero_grad();
}

void THSNN_Optimizer_zeroGrad(const Optimizer optimizer)
{
    (*optimizer)->zero_grad();
}

void THSNN_Optimizer_getParameters(const Optimizer optimizer, Tensor* (*allocator)(size_t length))
{
    auto parameters = (*optimizer)->parameters();
    Tensor * result = allocator(parameters.size());

    for (size_t i = 0; i < parameters.size(); i++)
    {
        result[i] = new torch::Tensor(parameters[i]);
    }
}

Tensor THSTorch_binary_cross_entropy(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction)
{
	CATCH_RETURN_TENSOR(torch::binary_cross_entropy(*input, *target, (weight == NULL ? at::Tensor() : *weight), reduction));
}

Tensor THSTorch_mse_loss(const Tensor input, const Tensor target, const int64_t reduction)
{
	CATCH_RETURN_TENSOR(torch::mse_loss(*input, *target, reduction));
}

Tensor THSTorch_nll_loss(const Tensor input, const Tensor target, const Tensor weight, const int64_t reduction)
{
	CATCH_RETURN_TENSOR(torch::nll_loss(*input, *target, (weight == NULL ? at::Tensor() : *weight), reduction));
}

Tensor THSTorch_poisson_nll_loss(const Tensor input, const Tensor target, const bool logInput, const bool full, const double eps, const int64_t reduction)
{
	CATCH_RETURN_TENSOR(torch::poisson_nll_loss(*input, *target, logInput, full, eps, reduction));
}

Optimizer THSNN_Adam_ctor(const Tensor* parameters, const int length, const double learnig_rate)
{
    auto  params = toTensors<at::Tensor>((torch::Tensor**)parameters, length);

    auto optimizer = torch::optim::Adam(params, learnig_rate);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::Adam>(torch::optim::Adam(params, learnig_rate)));
}

Optimizer THSNN_SGD_ctor(const Tensor* parameters, const int length, const double learnig_rate, const double momentum)
{
    auto  params = toTensors<at::Tensor>((torch::Tensor**)parameters, length);
    auto options = torch::optim::SGDOptions(learnig_rate).momentum(momentum);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::SGD>(torch::optim::SGD(params, options)));
}

void THSNN_Optimizer_step(const Optimizer optimizer)
{
    (*optimizer)->step();
}

void THSNN_initUniform(Tensor tensor, double low, double high)
{
    torch::nn::init::uniform_(*tensor, low, high);
}

// ########## To remove when updating to libtorch > 1.0.1 ############
enum class Nonlinearity {
    Linear,
    Conv1D,
    Conv2D,
    Conv3D,
    ConvTranspose1D,
    ConvTranspose2D,
    ConvTranspose3D,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU
};

enum class FanMode { FanIn, FanOut };

struct Fan {
    explicit Fan(torch::Tensor& tensor) {
        const auto dimensions = tensor.ndimension();
        TORCH_CHECK(
            dimensions >= 2,
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions");
        if (dimensions == 2) {
            in = tensor.size(1);
            out = tensor.size(0);
        }
        else {
            in = tensor.size(1) * tensor[0][0].numel();
            out = tensor.size(0) * tensor[0][0].numel();
        }
    }
    int64_t in;
    int64_t out;
};

double calculate_gain(Nonlinearity nonlinearity, double param) {
    if (nonlinearity == Nonlinearity::Tanh) {
        return 5.0 / 3.0;
    }
    else if (nonlinearity == Nonlinearity::ReLU) {
        return std::sqrt(2.0);
    }
    else if (nonlinearity == Nonlinearity::LeakyReLU) {
        return std::sqrt(2.0 / (1 + pow(param, 2)));
    }

    return 1.0;
}

double calculate_kaiming_std(
    Tensor tensor,
    double a,
    FanMode mode,
    Nonlinearity nonlinearity) {
    torch::NoGradGuard guard;
    Fan fan((*tensor));
    const auto gain = calculate_gain(nonlinearity, a);
    double std = 0.0;
    if (mode == FanMode::FanIn) {
        std = gain / std::sqrt(fan.in);
    }
    else {
        std = gain / std::sqrt(fan.out);
    }
    return std;
}

// ######################################################

void THSNN_initKaimingUniform(Tensor tensor, double a)
{
    //torch::nn::init::kaiming_uniform_(*tensor, a);
    // Since this is not available in PyTorch 1.0.1 will just used the original code for the moment
    auto std = calculate_kaiming_std(tensor, a, FanMode::FanIn, Nonlinearity::LeakyReLU);
    // Calculate uniform bounds from standard deviation
    const auto bound = std::sqrt(3.0) * std;
    tensor->uniform_(-bound, bound);
}

void THSNN_Optimizer_dispose(const Optimizer optimizer)
{
    delete optimizer;
}

void THSNN_Module_dispose(const NNModule module)
{
    delete module;
}

