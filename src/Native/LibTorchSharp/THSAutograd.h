// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "Utils.h"

// Returns whether the grad is enabled or not.
EXPORT_API(bool) THSAutograd_isGradEnabled();

// Enables / disables grad.
EXPORT_API(void) THSAutograd_setGrad(bool enabled);
