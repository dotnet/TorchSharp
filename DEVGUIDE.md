
# Building

    dotnet build /p:SkipNative=true
    dotnet build  # for cuda support on Windows and Linux
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

## Mac

Requirements:
- Clang/LLVM 12.0.0
- git
- .NET SDK 5.0.300
- Cmake 3.20.3

Build with

    dotnet build /p:SkipNative=true

## Packages

An ephemeral feed of packages from Azure DevOps CI is available for those 

* View link: https://dotnet.visualstudio.com/TorchSharp/_packaging?_a=feed&feed=SignedPackages
* Nuget feed: https://dotnet.pkgs.visualstudio.com/TorchSharp/_packaging/SignedPackages/nuget/v3/index.json

Some releases are pushed to nuget

## Building the TorchSharp package

    dotnet build
    dotnet pack

Locally built packages have names like this, names update every day.  If repeatedly rebuilding them locally you may have to remove them
from your local `.nuget` package cache.

    bin/packages/Debug/TorchSharp.0.3.0-local-Debug-20200520.nupkg
    bin/packages/Release/TorchSharp.0.3.0-local-Release-20200520.nupkg

To change the TorchSharp package version update this [file](https://github.com/dotnet/TorchSharp/blob/main/build/BranchInfo.props).

## Doing releases of the TorchSharp package

The TorchSharp package is pushed to nuget.org via Azure DevOps CI release pipeline.  Assuming you're not building or updating the LibTorch packages
(`BuildLibTorchPackages` is `false` in [azure-pipelines.yml](azure-pipelines.yml)) this is pretty simple once you have the permissions:

1. Update the version number in [./build/BranchInfo.props](./build/BranchInfo.props) and in the [Release Notes](./RELEASENOTES.md) file and then submit a PR. 

   Updating the major or minor version number should only be done after a discussion with repo admins. The patch number should be incremented by one each release and set to zero after a change to the major or minor version.
2. Integrate code to main and wait for CI to process
3. Go to [releases](https://donsyme.visualstudio.com/TorchSharp/_release) and choose "Create Release" (top right)
4. Under "Artifacts-->Version" choose the pipeline build corresponding to the thing you want to release.  It should be a successful build on main
5. Press "Create"

6. Once the package has been successfully pushed and is available in the NuGet gallery, create a GitHub tag in the 'main' branch with the version  as the name of the tag.

# The libtorch packages

The libtorch packages are huge (~3GB compressed combined for CUDA Windows) and cause a
lot of problems to make and deliver due to NuGet package size restrictions.

These problems include:

1. A massive 2GB binary in the linux CUDA package and multiple 1.0GB binaries in Windows CUDA package

2. Size limitations of about ~500MB on NuGet packages on the Azure DevOps CI system and about ~250MB on `nuget.org`

4. Regular download/upload failures on these systems due to network interruptions for packages of this size

5. 10GB VM image size restrictions for the containers userd to build these packages in the Azure DevOps CI system, we can easily run out of room.

6. Complete libtorch-cpu packages can't be built using your local machine alone, since they won't contain the
   full range of native bits. Instead they are built using Azure Pipelines by combining builds

For this reason, we do the following

1. The head, referenceable packages that deliver a functioning runtime are any of:

       libtorch-cpu
       libtorch-cuda-11.3-linux-x64
       libtorch-cuda-11.3-win-x64

2. These packages are combo packages that reference multiple parts.  The parts are **not** independently useful.
   Some parts deliver a single vast file via `primary` and `fragment` packages.  A build task is then used to "stitch" these files back together
   to one file on the target machine with a SHA check.  This is a hack but there is no other realistic way to deliver
   these vast files as packages (the alternative is to abandon packaging and require a manual
   install/detect/link of PyTorch CUDA on all downstream systems, whcih is extremely problematic
   for many practical reasons).

   For example, the CUDA package fragments are defined in [libtorch-cuda](src/Redist/libtorch-cuda-11.3/libtorch-cuda-11.3.proj). See more details later in this document.

3. The `libtorch-*` packages are built in Azure DevOps CI
   [using this build pipeline](https://donsyme.visualstudio.com/TorchSharp/_build?definitionId=1&_a=summary) but only in main
   branch and only when `BuildLibTorchPackages` is set to true in [azure-pipelines.yml](azure-pipelines.yml) in the main branch.
   You must currently manually edit this and submit to main to get new `libtorch-*` packages
   built.  Also increment `LibTorchPackageVersion` if necessary.  Do a push to main and the packages will build.  This process could be adjusted but at least gets us off the ground.

4. After a successful build, the `libtorch-*` packages can be trialled using the package feed from CI (see above).  When
   they are appropriate they can be  pushed to nuget using
   [this manually invoked release pipeline](https://donsyme.visualstudio.com/TorchSharp/_release?_a=releases&view=mine&definitionId=1) in
   Azure DevOps CI (so they don't have to be manually downloaded and pushed to `nuget.org`)

   a. [Go to release pipeline](https://donsyme.visualstudio.com/TorchSharp/_release?_a=releases&view=mine&definitionId=1)

   b. Press 'New Release'

   c. Select the successful main CI build that includes the `libtorch` packages, create the release and wait for it to finish. You should
      see `Initialize job`, `Download artifact - dotnet.TorchSharp - packages`, `NuGet push`, `Finalize Job` succeeded.

   d. All packages should now be pushed to `nuget.org` and will appear after indexing.

6. If updating libtorch packages, remember to delete all massive artifacts from Azure DevOps and reset this `BuildLibTorchPackages` in [azure-pipelines.yml](azure-pipelines.yml) in main branch.

### Updating PyTorch version for new libtorch packages

This project grabs LibTorch and makes a C API wrapper for it, then calls these from C#. When updating to a newer
version of PyTorch then quite a lot of careful work needs to be done.

0. Make sure you have plenty of disk space, e.g. 15GB

0. Clean and reset to main

       git checkout main
       git clean -xfd .

1. Familiarise yourself with download links. See https://pytorch.org/get-started/locally/ for download links.

   For example Linux, LibTorch 1.11.0 uses link

       https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.11.0%2Bcpu.zip

   The downloads are acquired automatically in the build process. To update the version, update these:

       <LibTorchVersion>1.11.0</LibTorchVersion>

2. Run these to test downloads and update SHA hashes for the various LibTorch downloads:

       dotnet build src\Redist\libtorch-cpu\libtorch-cpu.proj /p:UpdateSHA=true /p:TargetOS=linux /t:Build
       dotnet build src\Redist\libtorch-cpu\libtorch-cpu.proj /p:UpdateSHA=true /p:TargetOS=mac /t:Build
       dotnet build src\Redist\libtorch-cpu\libtorch-cpu.proj /p:UpdateSHA=true /p:TargetOS=windows /p:Configuration=Release /t:Build
       dotnet build src\Redist\libtorch-cpu\libtorch-cpu.proj /p:UpdateSHA=true /p:TargetOS=windows /p:Configuration=Debug /t:Build

       dotnet build src\Redist\libtorch-cuda-11.3\libtorch-cuda-11.3.proj /p:UpdateSHA=true /p:TargetOS=linux /t:Build
       dotnet build src\Redist\libtorch-cuda-11.3\libtorch-cuda-11.3.proj /p:UpdateSHA=true /p:TargetOS=windows /p:Configuration=Release /t:Build
       dotnet build src\Redist\libtorch-cuda-11.3\libtorch-cuda-11.3.proj /p:UpdateSHA=true /p:TargetOS=windows /p:Configuration=Debug /t:Build

   Each of these will take a **very very long time** depending on your broadband connection.  This can't currently be done in CI.

   At this point you must **very very carefully** update the `<File Include= ...` entries under src\Redist projects for
   [libtorch-cpu](src/Redist/libtorch-cpu/libtorch-cpu.proj) and [libtorch-cuda](src/Redist/libtorch-cuda-11.3/libtorch-cuda-11.3.proj).

   Check the contents of the unzip of the archive, e.g.

       bin\obj\x86.Debug\libtorch-cpu\libtorch-shared-with-deps-1.11.0\libtorch\lib

   You must also precisely refactor the CUDA binaries into multiple parts so each package ends up under ~300MB.

   For example, the following snippet spreads the `torch_cuda_cu.dll` binary file into four fragments. 

   ```xml
    <File Include= "libtorch\lib\torch_cuda_cu.dll"  PackageSuffix="part9-primary" FileUnstitchIndex="0" FileUnstitchStart="0" FileUnstitchSize="250000000" />
    <File Include= "libtorch\lib\torch_cuda_cu.dll"  PackageSuffix="part9-fragment1" FileUnstitchIndex="1" FileUnstitchStart="250000000" FileUnstitchSize="250000000" />
    <File Include= "libtorch\lib\torch_cuda_cu.dll"  PackageSuffix="part9-fragment2" FileUnstitchIndex="2" FileUnstitchStart="500000000" FileUnstitchSize="250000000" />
    <File Include= "libtorch\lib\torch_cuda_cu.dll"  PackageSuffix="part9-fragment3" FileUnstitchIndex="3" FileUnstitchStart="750000000" FileUnstitchSize="-1" />
   ```

   They must all be called either 'primary,' which should be the first fragment, or 'fragmentN' where 'N' is the ordinal number of the fragment, starting with '1'. The current logic allows for as many as 10 non-primary fragments. If more are needed, the code in [FileRestitcher.cs](pkg/FileRestitcher/FileRestitcher/FileRestitcher.cs) and [RestitchPackage.targets](pkg/common/RestitchPackage.targets) needs to be updated. Note that the size of each fragment is expressed in bytes, and that fragment start must be
   the sum of the size of all previous fragments. A '-1' should be used for the last fragment (and only for the last fragment): it means that the fragment size will be based on how much there is still left of the file.

   Because file sizes change from release to release, it may be necessary to add or remove fragments. When you add a fragment, you also need to add a corresponding project folder under the `pkg/` top-level folder. The process of doing so is copy-paste-rename of existing folders. The same goes for adding parts (whether fragmented or not): you should add a corresponding folder and project file. If you remove a fragment (or part), you should remove the corresponding folder, or CI will end up building empty packages.

3. Add the SHA files:

       git add src\Redist\libtorch-cpu\*.sha
       git add src\Redist\libtorch-cuda-11.3\*.sha

   After this you may as well submit to CI just to see what happens, though keep going with the other steps below as well.

4. Build the native and managed code without CUDA

       dotnet build /p:SkipCuda=true

   The first stage unzips the archives, then CMAKE is run.

   Unzipping the archives may take quite a while

   Note that things may have changed in the LibTorch header files, linking flags etc.  There is a CMakeLists.txt that acquires
   the cmake information delievered in the LibTorch download. It can be subtle.

   If the vxcproj for the native code gets configured by cmake then you should now be able to start developing the C++ code in Visual Studio. In order to get the correct environment variables and PATH, start VS from the command line, not from the Start menu:

       devenv TorchSharp.sln

   e.g. the vcxproj is created here:

       bin\obj\x64.Debug\Native\LibTorchSharp\LibTorchSharp.vcxproj

5. Similarly build the native code with CUDA

       dotnet build

6. You must also adjust the set of binaries referenced for tests, see various files under `tests` and `NativeAssemblyReference` in
`TorchSharp\Directory.Build.targets`.

7. Run tests

       dotnet build test -c Debug
       dotnet build test -c Release

8. Try building packages locally. The build (including CI) doesn't build `libtorch-*` packages by default, just the managed package. To
   get CI to build new `libtorch-*` packages update this version and set `BuildLibTorchPackages` in [azure-pipelines.yml](azure-pipelines.yml):

       <LibTorchPackageVersion>1.11.0.1</LibTorchPackageVersion>

       dotnet pack -c Debug /p:SkipCuda=true
       dotnet pack -c Release /p:SkipCuda=true
       dotnet pack -c Debug
       dotnet pack -c Release

9. Submit to CI and debug problems.

10. Remember to delete all massive artifacts from Azure DevOps and reset this `BuildLibTorchPackages` in in [azure-pipelines.yml](azure-pipelines.yml)


## Building with Visual Studio

In order for builds to work properly using Visual Studio 2019 or 2022, you must start VS from the 'x64 Native Tools Command Prompt for VS 2022' (or 2019) in order for the full environment to be set up correctly. Starting VS from the desktop or taskbar will not work properly.
