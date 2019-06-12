#include "THSData.h"

// Typedefs for the iterators.
typedef torch::data::DataLoader<
    std::remove_reference_t<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<torch::data::datasets::MNIST, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>&>, torch::data::samplers::SequentialSampler> MNISTTrain_t;

typedef torch::data::DataLoader<
    std::remove_reference_t<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<torch::data::datasets::MNIST, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>&>, torch::data::samplers::RandomSampler> MNISTTest_t;

// Load an MNIST dataset from a file
DatasetIteratorBase * THSData_loaderMNIST(
    const char* filename,
    int64_t batchSize,
    bool isTrain)
{
    torch::data::datasets::MNIST::Mode mode = torch::data::datasets::MNIST::Mode::kTrain;

    if (!isTrain)
    {
        mode = torch::data::datasets::MNIST::Mode::kTest;
    }

    auto dataset = torch::data::datasets::MNIST(filename, mode)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    size_t size = dataset.size().value();

    if (isTrain)
    {
        auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(dataset), batchSize);

        std::shared_ptr<MNISTTrain_t> shared = std::move(loader);

        return new DatasetIterator<MNISTTrain_t>(shared->begin(), size, shared);
    }
    else
    {
        auto loader = torch::data::make_data_loader(std::move(dataset), batchSize);

        std::shared_ptr<MNISTTest_t> shared = std::move(loader);

        return new DatasetIterator<MNISTTest_t>(shared->begin(), size, shared);
    }
}

size_t THSData_size(DatasetIteratorBase * iterator)
{
    return iterator->getSize();
}

bool THSData_moveNext(DatasetIteratorBase * iterator)
{
    bool result = iterator->moveNext();
    return result;
}

void THSData_current(DatasetIteratorBase * iterator, Tensor* data, Tensor* target)
{
    iterator->current(data, target);
}

void THSData_reset(DatasetIteratorBase * iterator)
{
    iterator->reset();
}

void THSData_dispose(DatasetIteratorBase * iterator)
{
    delete iterator;
}