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

NNModule THSNN_reluModule()
{
    return new std::shared_ptr<torch::nn::Module>(torch::nn::Functional(torch::relu).ptr());
}

NNModule THSNN_linearModule(const int64_t input_size, const int64_t output_size, const bool with_bias)
{
    auto options = torch::nn::LinearOptions(input_size, output_size).with_bias(with_bias);
    return new std::shared_ptr<torch::nn::Module>(torch::nn::Linear(options).ptr());
}

NNModule THSNN_conv2dModule(
    const int64_t inputChannel,
    const int64_t outputChannel,
    const int64_t kernelSize,
    const int64_t stride,
    const int64_t padding)
{
    auto options = torch::nn::Conv2dOptions(inputChannel, outputChannel, kernelSize)
        .stride(stride)
        .padding(padding);
    auto conv = torch::nn::Conv2d(options);

    return new std::shared_ptr<torch::nn::Module>(conv.ptr());

}

NNModule THSNN_new_module(const char ** names, at::Tensor ** parameters, const bool * require_grad, const int length)
{
    torch::nn::Module* module = (torch::nn::Module*)new ModuleWrapper(names, parameters, require_grad, length);
    return new std::shared_ptr<torch::nn::Module>(module);
}

int THSNN_has_parameter(const NNModule module, const char * name)
{
    return (*module)->named_parameters().contains(name);
}

Tensor THSNN_get_parameter(const NNModule module, const char * name)
{
    return new torch::Tensor(*(*module)->named_parameters().find(name));
}

void THSNN_get_parameters(
    const NNModule module,
    Tensor* (*allocator1)(size_t length))
{

    auto parameters = (*module)->named_parameters();
    Tensor * result1 = allocator1(parameters.size());

    for (int i = 0; i < parameters.size(); i++)
    {
        result1[i] = new torch::Tensor(parameters[i].value());
    }
}

void THSNN_get_named_parameters(
    const NNModule module,
    Tensor* (*allocator1)(size_t length),
    const char** (*allocator2)(size_t length))
{

    auto parameters = (*module)->named_parameters();
    Tensor * result1 = allocator1(parameters.size());
    const char ** result2 = allocator2(parameters.size());

    for (int i = 0; i < parameters.size(); i++)
    {
        result1[i] = new torch::Tensor(parameters[i].value());
        result2[i] = make_sharable_string(parameters[i].key());
    }
}

int THSNN_is_training(NNModule module)
{
    return (*module)->is_training();
}

void THSNN_train(NNModule module)
{
    (*module)->train();
}

void THSNN_eval(NNModule module)
{
    (*module)->eval();
}

long THSNN_getNumberOfChildren(const NNModule module)
{
    return (*module)->children().size();
}

const char * THSNN_getChildModuleName(const NNModule module, const int index)
{
    return make_sharable_string((*module)->children()[index]->name());
}

const char * THSNN_getModuleName(const NNModule module)
{
    return make_sharable_string((*module)->name());
}

Tensor THSNN_reluApply(const Tensor tensor)
{
    return new torch::Tensor(torch::relu(*tensor));
}

Tensor THSNN_maxPool2DApply(
    const Tensor tensor, 
    const int kernelSizeLength, 
    const int64_t* kernelSize,
    const int strideLength,
    const int64_t* stride)
{
    return new torch::Tensor(torch::max_pool2d(
        *tensor, 
        at::IntList(kernelSize, kernelSizeLength), 
        at::IntList(stride, strideLength)));
}

Tensor THSNN_adaptiveAvgPool2DApply(const Tensor tensor, const int length, const int64_t* outputSize)
{
    return new torch::Tensor(torch::adaptive_avg_pool2d(*tensor, at::IntList(outputSize, length)));
}

Tensor THSNN_avgPool2DApply(const Tensor tensor,
	const int kernelSizeLength,
	const int64_t* kernelSize,
	const int strideLength,
	const int64_t* stride) 
{
	return new torch::Tensor(torch::avg_pool2d(
		*tensor,
		at::IntList(kernelSize, kernelSizeLength),
		at::IntList(stride, strideLength)));
}

Tensor THSNN_logSoftMaxApply(const Tensor tensor, const int64_t dimension)
{
    return new torch::Tensor(torch::log_softmax(*tensor, dimension));
}

Tensor THSNN_featureDropoutApply(const Tensor tensor)
{
    return new torch::Tensor(torch::nn::FeatureDropout()->forward(*tensor));
}

Tensor THSNN_dropoutModuleApply(
    const Tensor tensor, 
    const double probability, 
    const bool isTraining)
{
    return new torch::Tensor(torch::dropout(*tensor, probability, isTraining));
}

Tensor THSNN_linearModuleApply(
    const NNModule module,
    const Tensor tensor)
{
    at::Tensor result = (*module)->as<torch::nn::Linear>()->forward(*tensor);

    return new torch::Tensor(result);
}

Tensor THSNN_conv2DModuleApply(
    const NNModule module,
    const Tensor tensor)
{
    at::Tensor result = (*module)->as<torch::nn::Conv2d>()->forward(*tensor);

    return new torch::Tensor(result);
}

int THSNN_linear_with_bias(const NNModule module)
{
    return (*module)->as<torch::nn::Linear>()->options.with_bias_;
}

Tensor THSNN_linear_get_bias(const NNModule module)
{
    auto linear_module = (*module)->as<torch::nn::Linear>();

    if (linear_module->options.with_bias_)
    {
        return new torch::Tensor(linear_module->bias);
    }
    return nullptr;
}

void THSNN_linear_set_bias(const NNModule module, Tensor tensor)
{
    auto linear_module = (*module)->as<torch::nn::Linear>();

    if (linear_module->options.with_bias_) {
        linear_module->bias = *tensor;
    }
}

Tensor THSNN_linear_get_weight(const NNModule module)
{
    return new torch::Tensor((*module)->as<torch::nn::Linear>()->weight);
}

void THSNN_linear_set_weight(const NNModule module, Tensor tensor)
{
    auto linear_module = (*module)->as<torch::nn::Linear>();

    linear_module->weight = *tensor;
}

void THSNN_moduleZeroGrad(const NNModule module)
{
    (*module)->zero_grad();
}

void THSNN_optimizerZeroGrad(const Optimizer optimizer)
{
    (*optimizer)->zero_grad();
}

void THSNN_optimizer_get_parameters(const Optimizer optimizer, Tensor* (*allocator)(size_t length))
{
    auto parameters = (*optimizer)->parameters();
    Tensor * result = allocator(parameters.size());

    for (int i = 0; i < parameters.size(); i++)
    {
        result[i] = new torch::Tensor(parameters[i]);
    }
}

Tensor THSNN_lossBCE(
    const Tensor input, 
    const Tensor target, 
    const Tensor weight, 
    const int64_t reduction)
{
    return weight == NULL ?
        new torch::Tensor(torch::binary_cross_entropy(*input, *target, {}, reduction)) :
        new torch::Tensor(torch::binary_cross_entropy(*input, *target, *weight, reduction));
}

Tensor THSNN_lossMSE(const Tensor input, const Tensor target, const int64_t reduction)
{
    return new torch::Tensor(torch::mse_loss(*input, *target, reduction));
}

Tensor THSNN_lossNLL(
    const Tensor input, 
    const Tensor target, 
    const Tensor weight, 
    const int64_t reduction)
{
    return weight == NULL ?
        new torch::Tensor(torch::nll_loss(*input, *target, {}, reduction)) :
        new torch::Tensor(torch::nll_loss(*input, *target, *weight, reduction));
}

Tensor THSNN_loss_poisson_nll(
    const Tensor input,
    const Tensor target,
    const bool logInput,
    const bool full,
    const double eps,
    const int64_t reduction)
{
    torch::Tensor loss;
    CATCH(
        if (logInput)
        {
            loss = torch::exp(*input) - (*target) * (*input);
        }
        else
        {
            loss = (*input) - (*target) * torch::log(*input + eps);
        }

        if (full)
        {
            auto mask = (*target) > 1;
            loss.masked_select(mask) += ((*target) * at::log(*target) - (*target) + 0.5 * at::log(2 * M_PI * (*target))).masked_select(mask);
        }
    )

    if (reduction == Reduction::None)
    {
        return new torch::Tensor(loss);
    }
    else if (reduction == Reduction::Mean)
    {
        return new torch::Tensor(torch::mean(loss));
    }
    else // (reduction == Reduction::Sum)
    {
        return new torch::Tensor(torch::sum(loss));
    }
}

Optimizer THSNN_optimizerAdam(const Tensor* parameters, const int length, const double learnig_rate)
{
    auto  params = toTensors<at::Tensor>((torch::Tensor**)parameters, length);

    auto optimizer = torch::optim::Adam(params, learnig_rate);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::Adam>(torch::optim::Adam(params, learnig_rate)));
}

Optimizer THSNN_optimizerSGD(const Tensor* parameters, const int length, const double learnig_rate, const double momentum)
{
    auto  params = toTensors<at::Tensor>((torch::Tensor**)parameters, length);
    auto options = torch::optim::SGDOptions(learnig_rate)
        .momentum(momentum);

    return new std::shared_ptr<torch::optim::Optimizer>(std::make_shared<torch::optim::SGD>(torch::optim::SGD(params, options)));
}

void THSNN_optimizerStep(const Optimizer optimizer)
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
        AT_CHECK(
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

void THSNN_optimizerDispose(const Optimizer optimizer)
{
    delete optimizer;
}

void THSNN_moduleDispose(const NNModule module)
{
    delete module;
}

