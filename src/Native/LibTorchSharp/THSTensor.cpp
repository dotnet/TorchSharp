#include "THSTensor.h"

Tensor THSTensor_zeros(
    const int64_t * sizes,
    const int length,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(device)
        .requires_grad(requires_grad);

    return new torch::Tensor(torch::zeros(at::IntList(sizes, length), options));
}

Tensor THSTensor_ones(
    const int64_t * sizes, 
    const int length, 
    const int8_t scalar_type, 
    const char * device, 
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(device)
        .requires_grad(requires_grad);

    return new torch::Tensor(torch::ones(at::IntList(sizes, length), options));
}

Tensor THSTensor_empty(
    const int64_t * sizes,
    const int length,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(device)
        .requires_grad(requires_grad);

    return new torch::Tensor(torch::empty(at::IntList(sizes, length), options));
}

Tensor THSTensor_new(
    void * data, 
    const int64_t * sizes, 
    const int szlength, 
    const int64_t * strides, 
    const int stlength, 
    int8_t scalar_type,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .is_variable(true)
        .requires_grad(requires_grad);

    return new torch::Tensor(torch::from_blob(data, at::IntList(sizes, szlength), at::IntList(strides, stlength), options));
}

Tensor THSTensor_newLong(
    int64_t * data,
    const int64_t * sizes,
    const int szlength,
    const int64_t * strides,
    const int stlength,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(at::kLong))
        .is_variable(true)
        .requires_grad(requires_grad);
    return new torch::Tensor(torch::from_blob(data, at::IntList(sizes, szlength), at::IntList(strides, stlength), options));
}

Tensor THSTensor_newByteScalar(char data, bool requires_grad)
{
    return new torch::Tensor(torch::tensor(data).set_requires_grad(requires_grad));
}

Tensor THSTensor_newShortScalar(short data, bool requires_grad)
{
    return new torch::Tensor(torch::tensor(data).set_requires_grad(requires_grad));
}

Tensor THSTensor_newIntScalar(int data, bool requires_grad)
{
    return new torch::Tensor(torch::tensor(data).set_requires_grad(requires_grad));
}

Tensor THSTensor_newLongScalar(int64_t data, bool requires_grad)
{
    return new torch::Tensor(torch::tensor(data).set_requires_grad(requires_grad));
}

Tensor THSTensor_newDoubleScalar(double data, bool requires_grad)
{
    return new torch::Tensor(torch::tensor(data).set_requires_grad(requires_grad));
}

Tensor THSTensor_newFloatScalar(float data, bool requires_grad)
{
    return new torch::Tensor(torch::tensor(data).set_requires_grad(requires_grad));
}

Tensor THSTensor_rand(
    const int64_t * sizes,
    const int length,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad)
{
    Tensor tensor;
    CATCH(
        auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(device)
        .requires_grad(requires_grad);

        tensor = new torch::Tensor(torch::rand(at::IntList(sizes, length), options));
    )
    return tensor;
}

Tensor THSTensor_randn(
    const int64_t * sizes, 
    const int length, 
    const int8_t scalar_type, 
    const char * device, 
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(device)
        .requires_grad(requires_grad);

    return new torch::Tensor(torch::randn(at::IntList(sizes, length), options));
}

Tensor THSTensor_sparse(
    Tensor indices,
    Tensor values,
    const int64_t * sizes,
    const int length,
    const int8_t scalar_type,
    const char * device,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(device)
        .requires_grad(requires_grad);

    auto i = torch::autograd::as_variable_ref(*indices).data();
    auto v = torch::autograd::as_variable_ref(*values).data();

    return new torch::Tensor(torch::sparse_coo_tensor(i, v, at::IntList(sizes, length), options));
}

int64_t THSTensor_ndimension(const Tensor tensor)
{
    return tensor->ndimension();
}

int64_t THSTensor_stride(const Tensor tensor, const int64_t dimension)
{
    return tensor->stride(dimension);
}

int64_t THSTensor_size(const Tensor tensor, const int64_t dimension)
{
    return tensor->size(dimension);
}

void THSTensor_dispose(const Tensor tensor)
{
    delete tensor;
}

void * THSTensor_data(const Tensor tensor)
{
    return tensor->data_ptr();
}

Scalar THSTensor_item(const Tensor tensor)
{
    return new torch::Scalar(tensor->item());
}

Tensor THSTensor_get1(const Tensor tensor, int64_t index)
{
    return new torch::Tensor((*tensor)[index]);
}

Tensor THSTensor_get2(const Tensor tensor, int64_t index1, int64_t index2)
{
    return new torch::Tensor((*tensor)[index1][index2]);
}

Tensor THSTensor_get3(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3)
{
    return new torch::Tensor((*tensor)[index1][index2][index3]);
}

void THSTensor_set1(const Tensor tensor, int64_t index, Scalar value)
{
    CATCH(
        (*tensor)[index] = *value;
    )
}

void THSTensor_set2(const Tensor tensor, int64_t index1, int64_t index2, Scalar value)
{
    (*tensor)[index1][index2] = *value;
}

void THSTensor_set3(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, Scalar value)
{
    (*tensor)[index1][index2][index3] = *value;
}

int8_t THSTensor_type(const Tensor tensor)
{
    return (int8_t)tensor->scalar_type();
}

Tensor THSTensor_to_type(const Tensor tensor, int8_t scalar_type)
{
    return new torch::Tensor(tensor->toType(at::ScalarType(scalar_type)));
}

const char* THSTensor_deviceType(const Tensor tensor)
{
    auto device = tensor->device();
    auto device_type = DeviceTypeName(device.type());

    std::transform(device_type.begin(), device_type.end(), device_type.begin(), ::tolower);

    return make_sharable_string(device_type);
}

bool THSTensor_requires_grad(const Tensor tensor)
{
    return tensor->requires_grad();
}

Tensor THSTensor_set_requires_grad(const Tensor tensor, const bool requires_grad)
{
   return new torch::Tensor(tensor->set_requires_grad(requires_grad));
}

int THSTensor_isSparse(const Tensor tensor)
{
    return tensor->is_sparse();
}

int THSTensor_isVariable(const Tensor tensor)
{
    return tensor->is_variable();
}

Tensor THSTensor_indices(Tensor tensor)
{
    return new torch::Tensor(tensor->_indices());
}

Tensor THSTensor_values(Tensor tensor)
{
    return new torch::Tensor(tensor->_values());
}

Tensor THSTensor_cpu(const Tensor tensor)
{
	return new torch::Tensor(tensor->cpu());
}

Tensor THSTensor_cuda(const Tensor tensor)
{
	return new torch::Tensor(tensor->cuda());
}

Tensor THSTensor_grad(const Tensor tensor)
{
    torch::Tensor grad = tensor->grad();
    return grad.defined() ? new torch::Tensor(grad) : NULL;
}

void THSTensor_backward(Tensor tensor)
{
    tensor->backward();
}

Tensor THSTensor_to_dense(Tensor tensor)
{
    return new torch::Tensor(tensor->to_dense());
}

Tensor THSTensor_cat(const Tensor* tensors, const int length, const int64_t dim)
{
    return new torch::Tensor(torch::cat(toTensors<at::Tensor>((torch::Tensor**)tensors, length), dim));
}

Tensor THSTensor_clone(const Tensor input)
{
    return new torch::Tensor(input->clone());
}

Tensor THSTensor_contiguous(const Tensor input)
{
    return new torch::Tensor(input->contiguous());
}

Tensor THSTensor_index_select(Tensor tensor, int64_t dimension, Tensor index)
{
    return new torch::Tensor(tensor->index_select(dimension, *index));
}

Tensor THSTensor_squeeze(Tensor tensor, int64_t dimension)
{
    return new torch::Tensor(tensor->squeeze(dimension));
}

Tensor THSTensor_reshape(const Tensor tensor, const int64_t * shape, const int length)
{
    return new torch::Tensor(tensor->reshape(at::IntList(shape, length)));
}

Tensor THSTensor_stack(const Tensor* tensors, const int length, const int64_t dim)
{
    return new torch::Tensor(torch::stack(toTensors<at::Tensor>((torch::Tensor**)tensors, length), dim));
}

Tensor THSTensor_t(const Tensor tensor)
{
    return new torch::Tensor(tensor->t());
}

Tensor THSTensor_transpose(const Tensor tensor, const int64_t dim1, const int64_t dim2)
{
    return new torch::Tensor(tensor->transpose(dim1, dim2));
}

void THSTensor_transpose_(const Tensor tensor, const int64_t dim1, const int64_t dim2)
{
    tensor->transpose_(dim1, dim2);
}

Tensor THSTensor_view(const Tensor tensor, const int64_t * shape, const int length)
{
    return new torch::Tensor(tensor->view(at::IntList(shape, length)));
}

Tensor THSTensor_add(const Tensor left, const int value, const Tensor right)
{
    return new torch::Tensor(left->add(*right, value));
}

Tensor THSTensor_add_(const Tensor left, const int value, const Tensor right)
{
    return new torch::Tensor(left->add_(*right, value));
}

Tensor THSTensor_addS(const Tensor left, const Scalar right)
{
    return new torch::Tensor(left->add(*right));
}

Tensor THSTensor_addbmm(
    const Tensor mat,
    const Tensor batch1,
    const Tensor batch2,
    const float beta,
    const float alpha)
{
    return new torch::Tensor(mat->addbmm(*batch1, *batch2, beta, alpha));
}

Tensor THSTensor_addmm(
    const Tensor mat,
    const Tensor mat1,
    const Tensor mat2,
    const float beta,
    const float alpha)
{
    return new torch::Tensor(mat->addmm(*mat1, *mat2, beta, alpha));
}

Tensor THSTensor_argmax(const Tensor tensor, const int64_t dimension, bool keepDim)
{
    return new torch::Tensor(tensor->argmax(dimension, keepDim));
}

Tensor THSTensor_baddbmm(
    const Tensor batch1,
    const Tensor batch2,
    const Tensor mat,
    const float beta,
    const float alpha)
{
    return new torch::Tensor(mat->baddbmm(*batch1, *batch2, beta, alpha));
}

Tensor THSTensor_bmm(const Tensor batch1, const Tensor batch2)
{
    return new torch::Tensor(batch1->bmm(*batch2));
}

Tensor THSTensor_clamp(const Tensor input, const Scalar min, const Scalar max)
{
    return new torch::Tensor(input->clamp(*min, *max));
}

Tensor THSTensor_div(const Tensor left, const Tensor right)
{
    return new torch::Tensor(left->div(*right));
}

void THSTensor_div_(const Tensor left, const Tensor right)
{
    left->div_(*right);
}

Tensor THSTensor_divS(const Tensor left, const Scalar right)
{
    return new torch::Tensor(left->div(*right));
}

Tensor THSTensor_eq(const Tensor left, const Tensor right)
{
    return new torch::Tensor(left->eq(*right));
}

Tensor THSTensor_eqS(const Tensor left, const Scalar right)
{
    return new torch::Tensor(left->eq(*right));
}

int THSTensor_equal(const Tensor left, const Tensor right)
{
    return left->equal(*right);
}

Tensor THSTensor_exp(const Tensor tensor)
{
    return new torch::Tensor(tensor->exp());
}

Tensor THSTensor_geS(const Tensor tensor, const Scalar scalar)
{
    return new torch::Tensor(tensor->ge(*scalar));
}

Tensor THSTensor_matmul(const Tensor left, const Tensor right)
{
    return  new torch::Tensor(left->matmul(*right));
}

void THSTensor_max(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dimension, const bool keep_dim)
{
    auto max = tensor->max(dimension, keep_dim);
    Tensor * result = allocator(2);
    result[0] = new torch::Tensor(std::get<0>(max));
    result[1] = new torch::Tensor(std::get<1>(max));
}

Tensor THSTensor_mean(const Tensor tensor)
{
    return new torch::Tensor(tensor->mean());
}

Tensor THSTensor_mm(const Tensor left, const Tensor right)
{
    return new torch::Tensor(left->mm(*right));
}

Tensor THSTensor_mul(const Tensor left, const Tensor right)
{
    return new torch::Tensor(left->mul(*right));
}

void THSTensor_mul_(const Tensor left, const Tensor right)
{
    left->mul_(*right);
}

Tensor THSTensor_mulS(const Tensor tensor, const Scalar scalar)
{
    return new torch::Tensor(tensor->mul(*scalar));
}

Tensor THSTensor_norm(const Tensor tensor, const int64_t dimension, const bool keep_dimension)
{
    return new torch::Tensor(tensor->norm(0, dimension, keep_dimension));
}

Tensor THSTensor_pow(const Tensor tensor, const Scalar scalar)
{
    return new torch::Tensor(tensor->pow(*scalar));
}

Tensor THSTensor_sigmoid(const Tensor tensor)
{
    return new torch::Tensor(tensor->sigmoid());
}

Tensor THSTensor_sub(const Tensor left, const Tensor right)
{
    return new torch::Tensor(left->sub(*right));
}

void THSTensor_sub_(const Tensor left, const Tensor right)
{
    left->sub_(*right);
}

Tensor THSTensor_sum(const Tensor tensor)
{
    return new torch::Tensor(tensor->sum());
}

Tensor THSTensor_sum1(const Tensor tensor, const int64_t * dimensions, int length, bool keep_dimension)
{
    return new torch::Tensor(tensor->sum(at::IntList(dimensions, length), keep_dimension));
}
