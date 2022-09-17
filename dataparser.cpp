#include <cstdio>
#include <ostream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <algorithm>
#include <random>
#include <fstream>

int main() {

    std::string inFilePath, outFilePath;
    char c;
    int n, blocks;

    std::cout << "Input file path: ";
    std::cin >> inFilePath;

    std::cout << "Output file path: ";
    std::cin >> outFilePath;

    std::cout << "Reset output file (y/n): ";
    std::cin >> c;

    std::cout << "N = ";
    std::cin >> n;
    std::cout << "Block count = ";
    std::cin >> blocks;

    if (c == 'y') {
        FILE *deleteFile;
        deleteFile = fopen(outFilePath.c_str(), "wb");
        fclose(deleteFile);
    }


    std::ifstream f(inFilePath, std::ios_base::in);

    auto rng = std::default_random_engine {};

    int blockSize = n / blocks;
    std::string line;
    for (int i = 0; i < blocks; i++) {

        std::vector<std::string> data;

        for (int j = 0; j < blockSize; j++) {
            if (std::getline(f, line)) {
                data.emplace_back(line);
            }
        }

        std::shuffle(data.begin(), data.end(), rng);

        FILE *out;
        out = fopen(outFilePath.c_str(), "awb");
        for (std::string entry : data) {
            int wdlIdx = entry.find_first_of('[');
            int scoreIdx = entry.find_first_of(']') + 2;
            std::string fen = entry.substr(0, wdlIdx);
            size_t fenSize = fen.size();
            std::string wdlStr = entry.substr(wdlIdx + 1, 3);
            int wdl = wdlStr == "0.0" ? 0 : (wdlStr == "0.5" ? 1 : 2);
            int score = std::stoi(entry.substr(scoreIdx, fenSize-scoreIdx+1));

            fwrite(&fenSize, sizeof(size_t), 1, out);
            fwrite(fen.c_str(), sizeof(char), fenSize, out);
            fwrite(&wdl, sizeof(int), 1, out);
            fwrite(&score, sizeof(int), 1, out);
        }

        fclose(out);
        std::cout << "Block " << i + 1 << " finished shuffling" << std::endl;
    }

    return 0;
}