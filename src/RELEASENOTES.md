## TorchSharp Release Notes

Releases, starting with 9/2/2021, are listed with the most recent release at the top.

### NuGet Version 0.92.52214

This was the first release since moving TorchSharp to the .NET Foundation organization. Most of the new functionality is related to continuing the API changes that were started in the previous release, and fixing some bugs.

__Fixed Bugs:__

#318 A few inconsistencies with the new naming

__Added Features:__

```
torch.nn.MultiHeadAttention
torch.linalg.cond
torch.linalg.cholesky_ex
torch.linalg.inv_ex
torch.amax/amin
torch.matrix_exp
torch.distributions.*   (about half the namespace)
```

__API Changes:__

CustomModule removed, its APIs moved to Module.
