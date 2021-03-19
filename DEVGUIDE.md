
# Building

    dotnet build
    dotnet build /p:SkipNative=true
    dotnet test
    dotnet pack

## Windows

Requirements:
- Visual Studio
- git
- cmake (tested with 3.18)

## Linux

Requirements:
- requirements to run .NET Core 3.1
- git
- cmake (tested with 3.14)
- clang 6.x +

Example to fulfill the requirements in Ubuntu 16:
```
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt-add-repository "deb https://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main"
sudo apt-get -y update
sudo apt-get -y install clang-6.0 git cmake libunwind8 curl libomp-dev
```

Commands:


## Packages

An ephemeral feed of packages from CI is available 

* View link: https://donsyme.visualstudio.com/TorchSharp/_packaging?_a=feed&feed=packages2
* Nuget feed: https://donsyme.pkgs.visualstudio.com/TorchSharp/_packaging/packages2/nuget/v3/index.json


## Building the TorchSharp package

    dotnet build
    dotnet pack

Locally built packages have names like this, names update every day.  If repeatedly rebuilding them locally you may have to remove them
from your local `.nuget` package cache.

    bin/packages/Debug/TorchSharp.0.3.0-local-Debug-20200520.nupkg
    bin/packages/Release/TorchSharp.0.3.0-local-Release-20200520.nupkg

To change the TorchSharp package version update this [file](https://github.com/xamarin/TorchSharp/blob/master/build/BranchInfo.props).

## Doing releases of the TorchSharp package

The TorchSharp package is pushed to nuget.org maually via Azure DevOps CI release pipeline, see below.

# The libtorch packages

The libtorch packages are huge (~3GB compressed combined for CUDA Windows) and cause a
lot of problems to make and deliver due to nuget package size restrictions.

These problems include:

1. A massive 2GB binary in the linux CUDA package and multiple 1.0GB binaries in Windows CUDA package

2. Size limitations of about ~500MB on nuget packages on the Azure DevOps CI system and about ~250MB on `nuget.org`

4. Regular download/upload failures on these systems due to network interruptions for packages of this size

5. 10GB VM image size restrictions for the containers userd to build these packages in the Azure DevOps CI system, we can easily run out of room.

6. Complete libtorch-cpu packages can't be built using your local machine alone, since they won't contain the
   full range of native bits. Instead they are built using Azure Pipelines by combining builds

For this reason, we do the following

1. The head, referenceable packages that deliver a functioning runtime are any of:

       libtorch-cpu
       libtorch-cuda-11.1-linux-x64
       libtorch-cuda-11.1-win-x64

2. These packages are combo packages that reference multiple parts.  The parts are **not** independently useful.
   Some parts deliver a single vast file via `primary` and `fragment` packages.  A build task is then used to "stitch" these files back together 
   to one file on the target machine with a SHA check.  This is a hack but there is no other realistic way to deliver
   these vast files as packages (the alternative is to abandon packaging and require a manual
   install/detect/link of PyTorch CUDA on all downstream systems, whcih is extremely problematic
   for many practical reasons).

3. The `libtorch-*` packages are built in Azure DevOps CI
   [using this build pipeline](https://donsyme.visualstudio.com/TorchSharp/_build?definitionId=1&_a=summary) but only in master
   branch and only when `BuildLibTorchPackages` is set to true in [azure-pipelines.yml](azure-pipelines.yml) in the master branch.
   You must currently manually edit this and submit to master to get new `libtorch-*` packages
   built.  Also increment `LibTorchPackageVersion` if necessary.  Do a push to master and the packages will build.  This process could be adjusted but at least gets us off the ground.

4. After a successful build, the `libtorch-*` packages can be trialled using the package feed from CI (see above).  When
   they are appropriate they can be  pushed to nuget using
   [this manually invoked release pipeline](https://donsyme.visualstudio.com/TorchSharp/_release?_a=releases&view=mine&definitionId=1) in
   Azure DevOps CI (so they don't have to be manually downloaded and pushed to `nuget.org`)

   a. [Go to release pipeline](https://donsyme.visualstudio.com/TorchSharp/_release?_a=releases&view=mine&definitionId=1)

   b. Press 'New Release'

   c. Select the successful master CI build that includes the `libtorch` packages, create the release and wait for it to finish. You should
      see `Initialize job`, `Download artifact - _xamarin.TorchSharp - packages`, `NuGet push`, `Finalize Job` succeeded.

   d. All packages should now be pushed to `nuget.org` and will appear after indexing.

6. If updating libtorch packages, remember to delete all massive artifacts from Azure DevOps and reset this `BuildLibTorchPackages` in [azure-pipelines.yml](azure-pipelines.yml) in master branch.

### Updating PyTorch version for new libtorch packages

This project grabs LibTorch and makes a C API wrapper for it, then calls these from C#. When updating to a newer
version of PyTorch then quite a lot of careful work needs to be done.

0. Make sure you have plenty of disk space, e.g. 15GB

1. Familiarise yourself with download links. See https://pytorch.org/get-started/locally/ for download links.

   For example Linux, LibTorch 1.8.0 uses link

       https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.8.0%2Bcpu.zip

   The downloads are acquired automatically in the build process. To update the version, update these:

       <LibTorchVersion>1.8.0</LibTorchVersion>

2. Run these to test downloads and update SHA hashes for the various LibTorch downloads:

       msbuild src\Redist\libtorch-cpu\libtorch-cpu.proj /p:UpdateSHA=true /p:TargetOS=linux /t:Build
       msbuild src\Redist\libtorch-cpu\libtorch-cpu.proj /p:UpdateSHA=true /p:TargetOS=windows /p:Configuration=Release /t:Build
       msbuild src\Redist\libtorch-cpu\libtorch-cpu.proj /p:UpdateSHA=true /p:TargetOS=windows /p:Configuration=Debug /t:Build

       msbuild src\Redist\libtorch-cuda-11.1\libtorch-cuda-11.1.proj /p:UpdateSHA=true /p:TargetOS=linux /t:Build
       msbuild src\Redist\libtorch-cuda-11.1\libtorch-cuda-11.1.proj /p:UpdateSHA=true /p:TargetOS=windows /p:Configuration=Release /t:Build
       msbuild src\Redist\libtorch-cuda-11.1\libtorch-cuda-11.1.proj /p:UpdateSHA=true /p:TargetOS=windows /p:Configuration=Debug /t:Build

   Each of these will take a **very very long time** depending on your broadband connection.  This can't currently be done in CI.

3. Add the SHA files:

       git add src\Redist\libtorch-cpu\*.sha
       git add src\Redist\libtorch-cuda-11.1\*.sha

   After this you may as well submit to CI just to see what happens, though keep going with the other steps below as well.

4. Build the native and managed code without CUDA

       dotnet build /p:SkipCuda=true

   The first stage unzips the archives, then CMAKE is run.

   Unzipping the archives may take quite a while

   Note that things may have changed in the LibTorch header files, linking flags etc.  There is a CMakeLists.txt that acquires
   the cmake information delievered in the LibTorch download. It can be subtle.
   If the vxcproj for the native code gets configured by cmake then you should now be able to start developing the C++ code in Visual Studio.

       devenv TorchSharp.sln

   e.g. the vcxproj is created here:
   
       bin\obj\x64.Debug\Native\LibTorchSharp\LibTorchSharp.vcxproj

5. Similarly build the native code with CUDA

       dotnet build

6. You must also **very very carefully** update the `<File Include= ...` entries under src\Redist projects for
   [libtorch-cpu](src\Redist\libtorch-cpu\libtorch-cpu.proj) and [libtorch-cuda](src\Redist\libtorch-cuda-11.1\libtorch-cuda-11.1.proj).
   
   Check the contents of the unzip of the archive, e.g.

       bin\obj\x86.Debug\libtorch-cpu\libtorch-shared-with-deps-1.8.0\libtorch\lib

   You must also precisely refactor the CUDA binaries into multiple parts so each package ends up under ~300MB.

7. You must also adjust the set of binaries referenced for tests, see various files under `tests` and `NativeAssemblyReference` in
`TorchSharp\Directory.Build.targets`.

8. Run tests

       dotnet build test -c Debug
       dotnet build test -c Release

9. Try building packages locally. The build (including CI) doesn't build `libtorch-*` packages by default, just the managed package. To
   get CI to build new `libtorch-*` packages update this version and set `BuildLibTorchPackages` in [azure-pipelines.yml](azure-pipelines.yml):

       <LibTorchPackageVersion>1.8.0.7</LibTorchPackageVersion>

       dotnet pack -c Debug /p:SkipCuda=true
       dotnet pack -c Release /p:SkipCuda=true
       dotnet pack -c Debug
       dotnet pack -c Release

10. Submit to CI and debug problems

11. Remember to delete all massive artifacts from Azure DevOps and reset this:

         <BuildLibTorchPackages>false</BuildLibTorchPackages>
