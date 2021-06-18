// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "THSTensor.h"

#include <iostream>
#include <fstream>

Tensor THSSpecial_entr(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::entr(*tensor));
}

Tensor THSSpecial_erf(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::erf(*tensor));
}

Tensor THSSpecial_erfc(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::erfc(*tensor));
}

Tensor THSSpecial_erfinv(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::erfinv(*tensor));
}

Tensor THSSpecial_expit(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::expit(*tensor));
}

Tensor THSSpecial_expm1(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::expm1(*tensor));
}

Tensor THSSpecial_exp2(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::exp2(*tensor));
}

Tensor THSSpecial_gammaln(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::gammaln(*tensor));
}

Tensor THSSpecial_i0e(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::i0e(*tensor));
}

Tensor THSSpecial_logit(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::logit(*tensor));
}

Tensor THSSpecial_xlog1py(const Tensor input, const Tensor other)
{
    CATCH_TENSOR(torch::special::xlog1py(*input, *other));
}
