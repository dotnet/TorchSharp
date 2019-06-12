#pragma once

#include "../Stdafx.h"

#include "Utils.h"

// Returns whether the grad is enabled or not.
EXPORT_API(bool) THSAutograd_isGradEnabled();

// Enables / disables grad.
EXPORT_API(void) THSAutograd_setGrad(bool enabled);
