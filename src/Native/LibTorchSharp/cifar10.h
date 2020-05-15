// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <c10/util/Optional.h>
#include <torch/types.h>

#include "torch/torch.h"

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <string>
#include <vector>

std::string join_paths(std::string head, const std::string& tail);
std::pair<at::Tensor, at::Tensor> read_dir(const std::string& root, bool train);
std::pair<at::Tensor, at::Tensor> read_cifar10(std::string path);

namespace torch {
    namespace data {
        namespace datasets {
            /// The CIFAR10 dataset.
            class CIFAR10 : public Dataset<CIFAR10> {
            public:
                /// The mode in which the dataset is loaded.
                enum class Mode { kTrain, kTest };

                /// Loads the CIFAR10 dataset from the `root` path.
                ///
                /// The supplied `root` path should contain the *content* of the unzipped
                /// CIFAR10 dataset, available from https://www.cs.toronto.edu/~kriz/cifar.html.
                explicit CIFAR10(const std::string& root, Mode mode = Mode::kTrain);

                /// Returns the `Example` at the given `index`.
                Example<> get(size_t index) override;

                /// Returns the size of the dataset.
                optional<size_t> size() const override;

                /// Returns true if this is the training subset of MNIST.
                bool is_train() const noexcept;

                /// Returns all images stacked into a single tensor.
                const Tensor& images() const;

                /// Returns all targets stacked into a single tensor.
                const Tensor& targets() const;

            private:
                Tensor images_, targets_;
                bool is_training;
            };
        } // namespace datasets
    } // namespace data
} // namespace torch
