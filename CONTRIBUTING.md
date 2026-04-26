Contributing to TorchSharp
==========================

If you are here, it means you are interested in helping us out. A hearty welcome and thank you! There are many ways you can contribute to the ML.NET project:

* Offer PR's to fix bugs or implement new features.
* Give us feedback and bug reports regarding the software or the documentation.
* Improve our examples, tutorials, and documentation.

This document describes contribution guidelines that are specific to TorchSharp. Please read [.NET Core Guidelines](https://github.com/dotnet/coreclr/blob/main/Documentation/project-docs/contributing.md) for more general .NET Core contribution guidelines.

## Developers

See the [Developer Guide](DEVGUIDE.md) for details about building and developing in this repo.


## Pull Requests

If you send us a PR, whether for documentation, examples, or library code, we require that you sign a digital Contributor License Agreement (CLA), so that we know you have the right to contribute. Once you have signed it, future PRs should go through without further requests to sign.

* **DO** use your own forked repository for all development, and submit cross-fork PRs.
* **DO** resolve merge conflicts early, by merging recent changes to 'main' into your development fork before submitting a PR.
* **DO** submit all code changes via pull requests (PRs). PRs will be reviewed and potentially merged by the repo maintainers after a peer review that includes at least one maintainer.
* **DO** give PRs short-but-descriptive names (for example, "Improve code coverage for System.Console by 10%", not "Fix #1234")
* **DO** refer to any relevant issues, and include [keywords](https://help.github.com/articles/closing-issues-via-commit-messages/) that automatically close issues when the PR is merged.
* **DO** tag any users that should know about and/or review the change.
* **DO** ensure each commit successfully builds.  The entire PR must pass all tests in the Continuous Integration (CI) system before it'll be merged.
* **DO** label your PR so it appears in the correct release-note category (see the PR template for the label table). Release notes are generated automatically — you do not need to edit `RELEASENOTES.md`.
* **DO** address PR feedback in an additional commit(s) rather than amending the existing commits, and only rebase/squash them when necessary.  This makes it easier for reviewers to track changes.
* **DO** assume that ["Squash and Merge"](https://github.com/blog/2141-squash-your-commits) will be used to merge your commit unless you request otherwise in the PR.
* **DO NOT** fix merge conflicts using a merge commit. Prefer `git rebase`.
* **DO NOT** mix independent, unrelated changes in one PR. Separate unrelated fixes into separate PRs.


## A Useful Tip

A useful tip from the Tensorflow.NET repo:

After you fork, add dotnet/TorchSharp as 'upstream' to your local repo ...

```git
git remote add upstream https://github.com/dotnet/TorchSharp.git
```

This makes it easy to keep your fork up to date by regularly pulling and merging from upstream.

Assuming that you do all your development off your main branch, keep your main updated
with these commands:

```git
git checkout main
git pull upstream main
git push origin main
```

Then, you merge onto your dev branch:

```git
git checkout <<your dev branch>>
git merge main
```
