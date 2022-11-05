#include <vector>
#include <cstring>
#include <string>
#include <cassert>
#include <thread>
#include <iostream>

constexpr bool WHITE = 0;
constexpr bool BLACK = 1;
constexpr unsigned int KING = 0;
constexpr unsigned int PAWN = 1;
constexpr unsigned int KNIGHT = 2;
constexpr unsigned int BISHOP = 3;
constexpr unsigned int ROOK = 4;
constexpr unsigned int QUEEN = 5;

unsigned int getOffset(const bool color, const unsigned int piece, const unsigned int square, const bool perspective) {
    return (perspective == WHITE ? color : !color) * 384 + piece * 64 + (perspective == WHITE ? square : square^56);
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
    bool* stm;
    bool* whiteFeatures;
    bool* blackFeatures;

    Batch(const std::vector<DataEntry> &data) {
        this->size = data.size();

        this->scores = new int[size];
        this->wdl = new int[size];
        this->stm = new bool[size];
        this->whiteFeatures = new bool[size * 768];
        this->blackFeatures = new bool[size * 768];

        std::memset(whiteFeatures, 0, size * 768);
        std::memset(blackFeatures, 0, size * 768);

        bool *whiteFeaturePointer = whiteFeatures;
        bool *blackFeaturePointer = blackFeatures;
        for (int i = 0; i < size; i++) {
            whiteFeaturePointer = whiteFeatures + i * 768;
            blackFeaturePointer = blackFeatures + i * 768;

            // Processing the fen
            unsigned int sq = 56, idx = 0;
            for (char c : data[i].fen) {
                idx++;
                if ('1' <= c && c <= '8') {
                    sq += c - '0';
                } else if (c == '/') {
                    sq -= 16;
                } else if (c == ' ') {
                    stm[i] = data[i].fen[idx] == 'w' ? 0 : 1;
                    break;
                } else {
                    switch (c) {
                        case 'k':
                            whiteFeaturePointer[getOffset(BLACK, KING, sq, WHITE)] = true;
                            blackFeaturePointer[getOffset(BLACK, KING, sq, BLACK)] = true;
                            break;
                        case 'K':
                            whiteFeaturePointer[getOffset(WHITE, KING, sq, WHITE)] = true;
                            blackFeaturePointer[getOffset(WHITE, KING, sq, BLACK)] = true;
                            break;
                        case 'p':
                            whiteFeaturePointer[getOffset(BLACK, PAWN, sq, WHITE)] = true;
                            blackFeaturePointer[getOffset(BLACK, PAWN, sq, BLACK)] = true;
                            break;
                        case 'P':
                            whiteFeaturePointer[getOffset(WHITE, PAWN, sq, WHITE)] = true;
                            blackFeaturePointer[getOffset(WHITE, PAWN, sq, BLACK)] = true;
                            break;
                        case 'n':
                            whiteFeaturePointer[getOffset(BLACK, KNIGHT, sq, WHITE)] = true;
                            blackFeaturePointer[getOffset(BLACK, KNIGHT, sq, BLACK)] = true;
                            break;
                        case 'N':
                            whiteFeaturePointer[getOffset(WHITE, KNIGHT, sq, WHITE)] = true;
                            blackFeaturePointer[getOffset(WHITE, KNIGHT, sq, BLACK)] = true;
                            break;
                        case 'b':
                            whiteFeaturePointer[getOffset(BLACK, BISHOP, sq, WHITE)] = true;
                            blackFeaturePointer[getOffset(BLACK, BISHOP, sq, BLACK)] = true;
                            break;
                        case 'B':
                            whiteFeaturePointer[getOffset(WHITE, BISHOP, sq, WHITE)] = true;
                            blackFeaturePointer[getOffset(WHITE, BISHOP, sq, BLACK)] = true;
                            break;
                        case 'r':
                            whiteFeaturePointer[getOffset(BLACK, ROOK, sq, WHITE)] = true;
                            blackFeaturePointer[getOffset(BLACK, ROOK, sq, BLACK)] = true;
                            break;
                        case 'R':
                            whiteFeaturePointer[getOffset(WHITE, ROOK, sq, WHITE)] = true;
                            blackFeaturePointer[getOffset(WHITE, ROOK, sq, BLACK)] = true;
                            break;
                        case 'q':
                            whiteFeaturePointer[getOffset(BLACK, QUEEN, sq, WHITE)] = true;
                            blackFeaturePointer[getOffset(BLACK, QUEEN, sq, BLACK)] = true;
                            break;
                        case 'Q':
                            whiteFeaturePointer[getOffset(WHITE, QUEEN, sq, WHITE)] = true;
                            blackFeaturePointer[getOffset(WHITE, QUEEN, sq, BLACK)] = true;
                            break;
                    }

                    sq++;
                }
            }

            scores[i] = stm[i]==0?data[i].score:-data[i].score;

            wdl[i] = data[i].wdl;
        }
    }

    ~Batch() {
        delete[] scores;
        delete[] wdl;
        delete[] stm;
        delete[] whiteFeatures;
        delete[] blackFeatures;
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
