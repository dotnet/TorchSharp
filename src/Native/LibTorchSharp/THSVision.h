// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"

// API

EXPORT_API(Tensor) THSVision_RGBtoHSV(const Tensor img, Tensor* s, Tensor *v);
EXPORT_API(Tensor) THSVision_HSVtoRGB(const Tensor h, const Tensor s, const Tensor v);
