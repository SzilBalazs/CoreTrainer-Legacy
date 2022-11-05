#include "json.hpp"
#include <fstream>
#include <cstdio>
#include <iostream>

using json = nlohmann::json;

constexpr int L_0_SIZE = 768;
constexpr int L_1_SIZE = 256;
constexpr int SCALE = 255;

int16_t L_0_WEIGHTS[L_0_SIZE * L_1_SIZE];
int16_t L_0_BIASES[L_1_SIZE];
int16_t L_1_WEIGHTS[L_1_SIZE * 2];
int16_t L_1_BIASES[1];

int main() {
    std::string inFile;
    std::cout << "Input file path: ";
    std::cin >> inFile;
    std::ifstream f(inFile);
    json data = json::parse(f);
    FILE *file;
    file = fopen("corenet.bin", "wb");

    for (int i = 0; i < L_1_SIZE; i++) {

        L_0_BIASES[i] = double(data["l0.bias"][i]) * SCALE;

        auto weights = data["l0.weight"][i];

        for (int j = 0; j < L_0_SIZE; j++) {
            L_0_WEIGHTS[j * L_1_SIZE + i] = double(weights[j]) * SCALE;
        }
    }

    for (int i = 0; i < L_1_SIZE * 2; i++) {
        L_1_WEIGHTS[i] = double(data["l1.weight"][0][i]) * SCALE;
    }

    L_1_BIASES[0] = double(data["l1.bias"][0]) * SCALE;

    fwrite(L_0_WEIGHTS, sizeof(int16_t), L_0_SIZE * L_1_SIZE, file);
    fwrite(L_0_BIASES, sizeof(int16_t), L_1_SIZE, file);

    fwrite(L_1_WEIGHTS, sizeof(int16_t), L_1_SIZE * 2, file);
    fwrite(L_1_BIASES, sizeof(int16_t), 1, file);

    fclose(file);

    return 0;
}
