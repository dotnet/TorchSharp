// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "Utils.h"

// Inter-op classes.

// Base non-generic interator class. Used to communicate with C#.
class DatasetIteratorBase
{
public:
    explicit
    DatasetIteratorBase() {}
    virtual size_t getSize() = 0;
    virtual bool moveNext() = 0;
    virtual void current(Tensor* data, Tensor* target) = 0;
    virtual void reset() = 0;
    virtual ~DatasetIteratorBase() {}
};

// Generic version of the iterator class.
template<typename Dataset>
class DatasetIterator : public DatasetIteratorBase
{
public:
    DatasetIterator(
        torch::data::Iterator<torch::data::Example<>> i,
        size_t s,
        std::shared_ptr<Dataset> l) : 
        DatasetIteratorBase(), 
        loaderPointer(l),
        currentIter(torch::data::Iterator<torch::data::Example<>>(i)),
        size(s) {}

        size_t getSize();
        bool moveNext();
        void current(Tensor* data, Tensor* target);
        void reset();

private:
    std::shared_ptr<Dataset> loaderPointer;
    torch::data::Iterator<torch::data::Example<>> currentIter;
    size_t size;
};

// Class-related methods.

// Get the total size in bytes of the input dataset.
template<typename Dataset>
inline size_t DatasetIterator<Dataset>::getSize()
{
    return size;
}

// Advance the iterator.
template<typename Dataset>
inline bool DatasetIterator<Dataset>::moveNext()
{
    ++currentIter;

    return currentIter != loaderPointer->end();
}

// Get the current object pointed by the iterator.
template<typename Dataset>
inline void DatasetIterator<Dataset>::current(Tensor* data, Tensor* target)
{
    data[0] = new torch::Tensor(currentIter->data);
    target[0] = new torch::Tensor(currentIter->target);
}

// Reset the iterator to start from the beginning.
template<typename Dataset>
inline void DatasetIterator<Dataset>::reset()
{
    currentIter = loaderPointer->begin();
}

// API.

// Load a MNIST dataset from a directory.
EXPORT_API(DatasetIteratorBase *) THSData_loaderMNIST(
    const char* filename,
    int64_t batchSize,
    bool isTrain);

// Load a MNIST dataset from a directory.
EXPORT_API(DatasetIteratorBase *) THSData_loaderCIFAR10(
    const char* filename,
    int64_t batchSize,
    bool isTrain);

// Gets the size in byte of some dataset wrapped as iterator.
EXPORT_API(size_t) THSData_size(DatasetIteratorBase * iterator);

// Advances the pointer of the target iterator.
EXPORT_API(bool) THSData_moveNext(DatasetIteratorBase * iterator);

// Gets the curret data and target tensors pointed by the iterator.
EXPORT_API(void) THSData_current(DatasetIteratorBase * iterator, Tensor* data, Tensor* target);

// Resets the iterator.
EXPORT_API(void) THSData_reset(DatasetIteratorBase * iterator);

// Disposes the iterator.
EXPORT_API(void) THSData_dispose(DatasetIteratorBase * iterator);
