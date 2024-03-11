#pragma once
#include <fstream>
#include "../eigen-3.4.0/unsupported/Eigen/CXX11/Tensor"
#include <iostream>
#include <vector>
#include <cstdint>

inline uint32_t EndianSwap(uint32_t a)
{
    return (a << 24) | ((a << 8) & 0x00ff0000) | ((a >> 8) & 0x0000ff00) | (a >> 24);
}

class MNISTData
{
private:
    Eigen::Tensor<double, 1> label_data;
    Eigen::Tensor<double, 2> image_data; // 2D Tensor for images
    uint32_t image_count;
    size_t image_count1;
    size_t label_count;
    //uint8_t *labels;
    //uint8_t *pixels;

public:
    MNISTData() : image_count(0) {}
    
    bool Load(bool isTraining)
    {
        // Set the expected image count
        image_count = isTraining ? 60000 : 10000;

        // Read labels
        const char *fileName_labels = isTraining ? "..\\mnist-datasets\\single-label.idx1-ubyte" : "..\\mnist-datasets\\single-label.idx1-ubyte";
        std::ifstream file(fileName_labels, std::ios::binary | std::ios::ate);
        if (!file)
        {
            std::cerr << "Could not open " << fileName_labels << " for reading.\n";
            return false;
        }

        std::streamsize fileSize = file.tellg();
        // std::cout << fileSize << std::endl;
        file.seekg(8, std::ios::beg);
        label_count = fileSize - 8;
        label_data.resize(label_count);

        // Read labels into a temporary buffer
        std::vector<uint8_t> buffer(fileSize - 8);
        if (!file.read(reinterpret_cast<char *>(buffer.data()), label_count))
        {
            std::cerr << "Error reading from " << fileName_labels << "\n";
            return false;
        }

        // Convert and copy labels to label_data
        for (size_t i = 0; i < label_count; ++i)
        {
            label_data(i) = static_cast<double>(buffer[i]);
        }

        file.close();

        // Read images
        const char *fileName_images = isTraining ? "..\\mnist-datasets\\single-image.idx3-ubyte" : "..\\mnist-datasets\\single-image.idx3-ubyte";
        std::ifstream imageFile(fileName_images, std::ios::binary | std::ios::ate);
        if (!imageFile)
        {
            std::cerr << "Could not open " << fileName_images << " for reading.\n";
            return false;
        }

        imageFile.seekg(4, std::ios::beg);

        // Read the image count
        imageFile.read(reinterpret_cast<char *>(&image_count), sizeof(image_count));

        image_count = EndianSwap(image_count); // Only swap if your system is little-endian
        // std::cout << image_count << std::endl;
        image_count1 = image_count;

        std::streamsize imageSize = imageFile.tellg();
        imageFile.seekg(16, std::ios::beg);

        // Assuming image_count is correctly determined from the file's header

        std::vector<uint8_t> image_data_raw(28 * 28 * image_count);
        if (!imageFile.read(reinterpret_cast<char *>(image_data_raw.data()), image_data_raw.size()))
        {
            std::cerr << "Error reading from " << fileName_images << "\n";
            return false;
        }
        imageFile.close();

        // Initialize and populate the Eigen tensor for image data

        image_data.resize(image_count, 28 * 28); // Correct tensor dimensions

        for (size_t img = 0; img < image_count; ++img)
        {
            for (size_t i = 0; i < 28; ++i)
            {
                for (size_t j = 0; j < 28; ++j)
                {
                    size_t index = img * 28 * 28 + i * 28 + j; // Calculate linear index
                    image_data(img, i * 28 + j) = static_cast<double>(image_data_raw[index]) / 255.0;
                }
            }
        }

        return true;
    }

    void WriteLabelsToFile() const
    {
        std::ofstream out("labels_out.txt");
        if (!out.is_open())
        {
            std::cerr << "Failed to open labels_out.txt for writing.\n";
            return;
        }
        out << 1 << std::endl;
        out << 10 << std::endl;

        for (int j = 0; j < label_count; ++j)
        {

            for (int i = 0; i < 10; ++i)
            {
                if (i == label_data(j))
                    out << 1 << std::endl;
                else
                    out << 0 << std::endl;
            }
        }

        out.close();
        std::cout << "Label data has been written to labels_out.txt\n";
    }

    void writeImageToFile() const
    {
        std::ofstream out("image_out.txt");
        if (!out.is_open())
        {
            std::cerr << "Failed to open image_out.txt for writing.\n";
            return;
        }
        out << 2 << std::endl;
        out << 28 << std::endl;
        out << 28 << std::endl;

        for (size_t i = 0; i < image_count1; ++i)
        {
            for (size_t j = 0; j < 784; ++j)
            {
                out << image_data(i, j) << std::endl;
            }
        }

        out.close();
        std::cout << "Image data has been written to image_out.txt\n";
    }

    size_t NumImages() const { return image_count; }

    /*const Eigen::Tensor<double, 2> GetImage(size_t index) const
    {
        // Calculate the start of the image data for the given index
        size_t start = index * 28 * 28;
        // Create a 2D tensor to hold the image data
        Eigen::Tensor<double, 2> image(28, 28);
        // Copy the data from the main image tensor to the image tensor
        for (size_t i = 0; i < 28; ++i)
        {
            for (size_t j = 0; j < 28; ++j)
            {
                image(i, j) = image_data(i * image_count + index, j);
            }
        }
        return image;
    }*/
};
