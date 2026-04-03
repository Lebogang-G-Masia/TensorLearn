#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

namespace MNIST {
    
    inline uint32_t swap_endian(uint32_t val) {
        val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
        return (val << 16) | (val >> 16);
    }

    struct Dataset {
        std::vector<std::vector<float>> images;
        std::vector<uint8_t> labels;
    };

    inline Dataset load(const std::string& img_path, const std::string& lbl_path, int limit = -1) {
        std::ifstream img_file(img_path, std::ios::binary);
        std::ifstream lbl_file(lbl_path, std::ios::binary);

        if (!img_file.is_open() || !lbl_file.is_open()) {
            throw std::runtime_error("Could not open MNIST files.");
        }

        uint32_t magic, num_items, rows, cols;
        img_file.read(reinterpret_cast<char*>(&magic), 4);
        img_file.read(reinterpret_cast<char*>(&num_items), 4);
        img_file.read(reinterpret_cast<char*>(&rows), 4);
        img_file.read(reinterpret_cast<char*>(&cols), 4);

        num_items = swap_endian(num_items);
        rows = swap_endian(rows);
        cols = swap_endian(cols);

        uint32_t magic_lbl, num_items_lbl;
        lbl_file.read(reinterpret_cast<char*>(&magic_lbl), 4);
        lbl_file.read(reinterpret_cast<char*>(&num_items_lbl), 4);
        num_items_lbl = swap_endian(num_items_lbl);

        if (num_items != num_items_lbl) throw std::runtime_error("MNIST image/label count mismatch.");

        int to_load = (limit > 0 && limit < (int)num_items) ? limit : num_items;
        Dataset ds;
        ds.images.resize(to_load, std::vector<float>(rows * cols));
        ds.labels.resize(to_load);

        for (int i = 0; i < to_load; i++) {
            for (uint32_t j = 0; j < rows * cols; j++) {
                unsigned char pixel = 0;
                img_file.read(reinterpret_cast<char*>(&pixel), 1);
                ds.images[i][j] = (float)pixel / 255.0f; // Normalize
            }
            lbl_file.read(reinterpret_cast<char*>(&ds.labels[i]), 1);
        }

        return ds;
    }
}

#endif
