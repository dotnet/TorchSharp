// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "THSAutograd.h"

#include "torch/torch.h"

bool THSAutograd_isGradEnabled()
{
    bool result = torch::autograd::GradMode::is_enabled();
    return result;
}

void THSAutograd_setGrad(bool enabled)
{
    torch::autograd::GradMode::set_enabled(enabled);
}