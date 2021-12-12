#include <torch/torch.h>

#include <iostream>

// Where to find the MNIST dataset.
const char* kDataRoot = "../data/MNIST/raw";

const int64_t kTrainBatchSize = 64;
const int64_t kTestBatchSize = 64;

template <typename DataLoader>
void move_data_to_device(torch::Device device, DataLoader& data_loader) {
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device);
    auto targets = batch.data.to(device);
    std::cout << "Successfully moved data to device:" << device << std::endl;
    return;
  }
}

auto main() -> int {
  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Using GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Using CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Stack<>());
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  move_data_to_device(device, *train_loader);
  move_data_to_device(device, *test_loader);

}
