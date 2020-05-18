// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "cifar10.h"

#include <cstddef>
#include <fstream>

const std::string kTrainImagesTargetsFilename[] = {
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin" };
constexpr const char* kTestImagesTargetsFilename = "test_batch.bin";

constexpr uint32_t kWidth = 32;
constexpr uint32_t kHight = 32;
constexpr uint32_t kChannel = 3;
constexpr uint64_t kBytesPerImage = kWidth * kHight * kChannel + 1;
constexpr uint32_t kImagesPerFile = 10000;

std::string join_paths(std::string head, const std::string& tail) {
    if (head.back() != '/') {
        head.push_back('/');
    }
    head += tail;
    return head;
}

std::pair<at::Tensor, at::Tensor> read_dir(const std::string& root, bool train) {
    if (train) {
        std::vector<std::pair<at::Tensor, at::Tensor>> images_targets;
        std::vector<at::Tensor> images;
        std::vector<at::Tensor> targets;
        for (auto path : kTrainImagesTargetsFilename) {
            const std::string completePath = join_paths(root, path);
            images_targets.push_back(read_cifar10(completePath));
        }

        for (auto pair : images_targets) {
            images.push_back(pair.first);
            targets.push_back(pair.second);
        }

        return std::pair<at::Tensor, at::Tensor>(torch::cat(images, 0), torch::cat(targets, 0));
    }

    auto path = join_paths(root, kTestImagesTargetsFilename);
    return read_cifar10(path);
}

std::pair<at::Tensor, at::Tensor> read_cifar10(std::string path) {
    std::ifstream data(path, std::ios::binary);
    TORCH_CHECK(data, "Error opening data file at ", path);
    auto content = torch::empty(kImagesPerFile * kBytesPerImage, torch::kByte);
    at::Tensor images = torch::zeros({ kImagesPerFile, kChannel, kHight, kWidth }, torch::kFloat32);
    at::Tensor labels = torch::zeros({ kImagesPerFile, }, torch::kInt64);

    data.read(reinterpret_cast<char*>(content.data_ptr()), content.numel());

    for (uint32_t i = 0; i < kImagesPerFile; i++)
    {
        auto offset = kBytesPerImage * i;
        labels.narrow(0, i, 1).copy_(content.narrow(0, offset, 1));
        images.narrow(0, i, 1).copy_(content
            .narrow(0, 1 + offset, kBytesPerImage - 1)
            .view({ 1, kChannel, kHight, kWidth })
            .toType(torch::kFloat32));
    }
    return std::pair<at::Tensor, at::Tensor>(images.div_(255.0), labels);
}

namespace torch {
    namespace data {
        namespace datasets {
            CIFAR10::CIFAR10(const std::string& root, Mode mode) {
                is_training = mode == Mode::kTrain;
                auto images_targets = read_dir(root, is_training);
                images_ = images_targets.first;
                targets_ = images_targets.second;
            }

            Example<> CIFAR10::get(size_t index) {
                return { images_[index], targets_[index] };
            }

            optional<size_t> CIFAR10::size() const {
                return images_.size(0);
            }

            bool CIFAR10::is_train() const noexcept {
                return is_training;
            }

            const Tensor& CIFAR10::images() const {
                return images_;
            }

            const Tensor& CIFAR10::targets() const {
                return targets_;
            }
        } // namespace datasets
    } // namespace data
} // namespace torch
