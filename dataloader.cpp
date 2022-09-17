#include <vector>
#include <cstring>
#include <string>
#include <cassert>
#include <iostream>

constexpr unsigned int WHITE = 0;
constexpr unsigned int BLACK = 1;
constexpr unsigned int KING = 0;
constexpr unsigned int PAWN = 1;
constexpr unsigned int KNIGHT = 2;
constexpr unsigned int BISHOP = 3;
constexpr unsigned int ROOK = 4;
constexpr unsigned int QUEEN = 5;

unsigned int getOffset(const unsigned int color, const unsigned int piece) {
    return color * 384 + piece * 64;
}

struct DataEntry {

    std::string fen;
    int score;
    int wdl;

    DataEntry(std::string &fen, int score, int wdl) {
        this->fen = fen;
        this->score = score;
        this->wdl = wdl;
    }
};

struct Batch {

    int size;
    int* scores;
    int* wdl;
    bool* features;

    Batch(const std::vector<DataEntry> &data) {
        this->size = data.size();

        this->scores = new int[size];
        this->wdl = new int[size];
        this->features = new bool[size * 768];

        std::memset(features, 0, size * 768);

        bool *featurePointer = features;
        for (int i = 0; i < size; i++) {
            featurePointer = features + i * 768;

            // Processing the fen
            unsigned int sq = 56, idx = 0;
            for (char c : data[i].fen) {
                idx++;
                if ('1' <= c && c <= '8') {
                    sq += c - '0';
                } else if (c == '/') {
                    sq -= 16;
                } else if (c == ' ') {
                    break;
                } else {
                    switch (c) {
                        case 'k':
                            featurePointer[getOffset(BLACK, KING) + sq] = true;
                            break;
                        case 'K':
                            featurePointer[getOffset(WHITE, KING) + sq] = true;
                            break;
                        case 'p':
                            featurePointer[getOffset(BLACK, PAWN) + sq] = true;
                            break;
                        case 'P':
                            featurePointer[getOffset(WHITE, PAWN) + sq] = true;
                            break;
                        case 'n':
                            featurePointer[getOffset(BLACK, KNIGHT) + sq] = true;
                            break;
                        case 'N':
                            featurePointer[getOffset(WHITE, KNIGHT) + sq] = true;
                            break;
                        case 'b':
                            featurePointer[getOffset(BLACK, BISHOP) + sq] = true;
                            break;
                        case 'B':
                            featurePointer[getOffset(WHITE, BISHOP) + sq] = true;
                            break;
                        case 'r':
                            featurePointer[getOffset(BLACK, ROOK) + sq] = true;
                            break;
                        case 'R':
                            featurePointer[getOffset(WHITE, ROOK) + sq] = true;
                            break;
                        case 'q':
                            featurePointer[getOffset(BLACK, QUEEN) + sq] = true;
                            break;
                        case 'Q':
                            featurePointer[getOffset(WHITE, QUEEN) + sq] = true;
                            break;
                    }

                    sq++;
                }
            }

            scores[i] = data[i].score;

            wdl[i] = data[i].wdl;
        }
    }

    ~Batch() {
        delete[] scores;
        delete[] features;
        delete[] wdl;
    }
};

struct BatchReader {

    FILE *file;
    unsigned int batchSize;
    unsigned int epoch;

    BatchReader(const char *filename, unsigned int batchSize) {
        this->file = fopen(filename, "rb");
        this->batchSize = batchSize;
        this->epoch = 1;

        assert(this->file != nullptr);
    }

    DataEntry readEntry() {
        size_t fenSize;
        fread(&fenSize, sizeof(size_t), 1, file);

        if (feof(file)) {
            rewind(file);
            epoch++;
            fread(&fenSize, sizeof(size_t), 1, file);
        }

        char *buffer = new char[fenSize];
        fread(buffer, sizeof(char), fenSize, file);

        std::string fen = {buffer};

        delete[] buffer;

        int wdl;
        fread(&wdl, sizeof(int), 1, file);

        int score;
        fread(&score, sizeof(int), 1, file);

        return DataEntry(fen, score, wdl);
    }

    Batch *readBatch() {
        std::vector<DataEntry> data;

        for (unsigned int i = 0; i < batchSize; i++) {

            data.emplace_back(readEntry());
        }

        return new Batch(data);
    }

    ~BatchReader() {
        fclose(file);
    }
};

extern "C" {

    BatchReader *newBatchReader(const char *filename, unsigned int batchSize) {
        return new BatchReader(filename, batchSize);
    }

    void deleteBatchReader(BatchReader *reader) {
        delete reader;
    }

    Batch* getBatch(BatchReader *reader) {
        return reader->readBatch();
    }

    void deleteBatch(Batch *batch) {
        delete batch;
    }
}
