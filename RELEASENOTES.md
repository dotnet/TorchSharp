## TorchSharp Release Notes

Releases, starting with 9/2/2021, are listed with the most recent release at the top.

### NuGet Version 0.93.0

With this release, releases will have explicit control over the patch version number.

__Fixed Bugs:__

Fixed incorrectly implemented Module APIs related to parameter / module registration.
Changed Module.state_dict() and Module.load() to 'virtual,' so that saving and restoring state may be customized.
#353 Missing torch.minimum (with an alternative raising exception)
#327 Tensor.Data<T> should do a type check

__API Changes:__

Removed the type-named tensor factories, such as 'Int32Tensor.rand(),' etc.

### NuGet Version 0.92.52220

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
