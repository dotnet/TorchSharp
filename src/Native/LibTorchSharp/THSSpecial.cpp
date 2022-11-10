// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSTensor.h"

#include <iostream>
#include <fstream>

Tensor THSSpecial_airy_ai(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::airy_ai(*tensor));
}

Tensor THSSpecial_airy_ai_out(const Tensor tensor, Tensor out)
{
    CATCH_TENSOR(torch::special::airy_ai_out(*out, *tensor));
}

Tensor THSSpecial_bessel_j0(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::bessel_j0(*tensor));
}

Tensor THSSpecial_bessel_j0_out(const Tensor tensor, Tensor out)
{
    CATCH_TENSOR(torch::special::bessel_j0_out(*out, *tensor));
}

Tensor THSSpecial_bessel_j1(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::bessel_j1(*tensor));
}

Tensor THSSpecial_bessel_j1_out(const Tensor tensor, Tensor out)
{
    CATCH_TENSOR(torch::special::bessel_j1_out(*out, *tensor));
}

Tensor THSSpecial_bessel_y0(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::bessel_y0(*tensor));
}

Tensor THSSpecial_bessel_y0_out(const Tensor tensor, Tensor out)
{
    CATCH_TENSOR(torch::special::bessel_y0_out(*out, *tensor));
}

Tensor THSSpecial_bessel_y1(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::bessel_y1(*tensor));
}

Tensor THSSpecial_bessel_y1_out(const Tensor tensor, Tensor out)
{
    CATCH_TENSOR(torch::special::bessel_y1_out(*out, *tensor));
}

Tensor THSSpecial_modified_bessel_i0(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::modified_bessel_i0(*tensor));
}

Tensor THSSpecial_modified_bessel_i0_out(const Tensor tensor, Tensor out)
{
    CATCH_TENSOR(torch::special::modified_bessel_i0_out(*out, *tensor));
}

Tensor THSSpecial_modified_bessel_i1(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::modified_bessel_i1(*tensor));
}

Tensor THSSpecial_modified_bessel_i1_out(const Tensor tensor, Tensor out)
{
    CATCH_TENSOR(torch::special::modified_bessel_i1_out(*out, *tensor));
}

Tensor THSSpecial_modified_bessel_k0(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::modified_bessel_k0(*tensor));
}

Tensor THSSpecial_modified_bessel_k0_out(const Tensor tensor, Tensor out)
{
    CATCH_TENSOR(torch::special::modified_bessel_k0_out(*out, *tensor));
}

Tensor THSSpecial_modified_bessel_k1(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::modified_bessel_k1(*tensor));
}

Tensor THSSpecial_modified_bessel_k1_out(const Tensor tensor, Tensor out)
{
    CATCH_TENSOR(torch::special::modified_bessel_k1_out(*out, *tensor));
}

Tensor THSSpecial_scaled_modified_bessel_k0(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::scaled_modified_bessel_k0(*tensor));
}

Tensor THSSpecial_scaled_modified_bessel_k0_out(const Tensor tensor, Tensor out)
{
    CATCH_TENSOR(torch::special::scaled_modified_bessel_k0_out(*out, *tensor));
}

Tensor THSSpecial_scaled_modified_bessel_k1(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::scaled_modified_bessel_k1(*tensor));
}

Tensor THSSpecial_scaled_modified_bessel_k1_out(const Tensor tensor, Tensor out)
{
    CATCH_TENSOR(torch::special::scaled_modified_bessel_k1_out(*out, *tensor));
}

Tensor THSSpecial_spherical_bessel_j0(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::spherical_bessel_j0(*tensor));
}

Tensor THSSpecial_spherical_bessel_j0_out(const Tensor tensor, Tensor out)
{
    CATCH_TENSOR(torch::special::spherical_bessel_j0_out(*out, *tensor));
}

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

Tensor THSSpecial_erfcx(const Tensor tensor)
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

Tensor THSSpecial_gammainc(const Tensor input, const Tensor other)
{
    CATCH_TENSOR(torch::special::gammainc(*input, *other));
}

Tensor THSSpecial_gammaincc(const Tensor input, const Tensor other)
{
    CATCH_TENSOR(torch::special::gammaincc(*input, *other));
}

Tensor THSSpecial_polygamma(const int64_t n, const Tensor tensor)
{
    CATCH_TENSOR(torch::special::polygamma(n, *tensor));
}

Tensor THSSpecial_digamma(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::digamma(*tensor));
}

Tensor THSSpecial_multigammaln(const Tensor tensor, const int64_t p)
{
    CATCH_TENSOR(torch::special::multigammaln(*tensor, p));
}

Tensor THSSpecial_i0e(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::i0e(*tensor));
}

Tensor THSSpecial_i0(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::i0(*tensor));
}

Tensor THSSpecial_i1e(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::i1e(*tensor));
}

Tensor THSSpecial_i1(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::i1(*tensor));
}

Tensor THSSpecial_logit(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::logit(*tensor));
}

Tensor THSSpecial_log_softmax(const Tensor tensor, int64_t dim, int8_t scalar_type)
{
    CATCH_TENSOR(torch::special::log_softmax(*tensor, dim, c10::ScalarType(scalar_type)));
}

Tensor THSSpecial_ndtr(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::ndtr(*tensor));
}

Tensor THSSpecial_ndtri(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::ndtri(*tensor));
}

Tensor THSSpecial_sinc(const Tensor tensor)
{
    CATCH_TENSOR(torch::special::sinc(*tensor));
}

Tensor THSSpecial_xlog1py(const Tensor input, const Tensor other)
{
    CATCH_TENSOR(torch::special::xlog1py(*input, *other));
}

Tensor THSSpecial_zeta(const Tensor input, const Tensor other)
{
    CATCH_TENSOR(torch::special::zeta(*input, *other));
}