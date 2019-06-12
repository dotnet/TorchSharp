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
