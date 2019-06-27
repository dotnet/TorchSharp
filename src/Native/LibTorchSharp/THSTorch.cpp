#include "THSTorch.h"

#include "torch/torch.h"

void THSTorch_seed(const int64_t seed)
{
    torch::manual_seed(seed);
}

int THSTorch_isCudaAvailable()
{
    return torch::cuda::is_available();
}

const char * THSTorch_get_and_reset_last_err()
{
    char *tmp = torch_last_err;
    torch_last_err = nullptr;
    return tmp;
}

Scalar THSTorch_btos(char value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_stos(short value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_itos(int value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_ltos(long value)
{
    return new torch::Scalar(int64_t(value));
}

Scalar THSTorch_ftos(float value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_dtos(double value)
{
    return new torch::Scalar(value);
}

void THSThorch_dispose_scalar(Scalar scalar)
{
    delete scalar;
}
