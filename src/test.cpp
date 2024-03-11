#include "data_loader.h"

int main(int argc, char **argv)
{
    MNISTData mnistData;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " --load-and-write" << std::endl;
        return 1;
    }

    std::string command = argv[1];

    if (command == "--load-and-write") {
        if (!mnistData.Load(true)) {
            std::cerr << "Could not load the training data!" << std::endl;
            return 1;
        }
        if (!mnistData.Load(false)) {
            std::cerr << "Could not load the test data!" << std::endl;
            return 1;
        }

        std::cout << "Loading the MNIST data is successful" << std::endl;
        std::cout << "The number of training images is: " << mnistData.NumImages() << std::endl;

        mnistData.writeImageToFile(); // Writes images to a file
        mnistData.WriteLabelsToFile(); // Writes labels to a file
    } else {
        std::cerr << "Invalid command." << std::endl;
        return 1;
    }

    return 0;
}
