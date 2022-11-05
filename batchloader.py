import os
import logging
import ctypes
import numpy as np
import torch

cwd = os.getcwd()
lib = ctypes.cdll.LoadLibrary(os.path.join(cwd, "dataloader.so"))


class Batch(ctypes.Structure):
    _fields_ = [
        ('size', ctypes.c_int),
        ('scores', ctypes.POINTER(ctypes.c_int)),
        ('wdl', ctypes.POINTER(ctypes.c_int)),
        ('stm', ctypes.POINTER(ctypes.c_bool)),
        ('whiteFeatures', ctypes.POINTER(ctypes.c_bool)),
        ('blackFeatures', ctypes.POINTER(ctypes.c_bool)),
    ]

    def get_tensors(self):
        score_tensor = torch.from_numpy(np.ctypeslib.as_array(self.scores, shape=(self.size, 1)).astype("float32"))
        stm_tensor = torch.from_numpy(np.ctypeslib.as_array(self.stm, shape=(self.size, 1)).astype("float32"))

        white_all_features = np.ctypeslib.as_array(self.whiteFeatures, shape=(self.size * 768, 1)).astype("float32")
        white_feature_tensor = torch.from_numpy(np.reshape(white_all_features, (-1, 768)))

        black_all_features = np.ctypeslib.as_array(self.blackFeatures, shape=(self.size * 768, 1)).astype("float32")
        black_feature_tensor = torch.from_numpy(np.reshape(black_all_features, (-1, 768)))

        wdl_tensor = torch.from_numpy(np.ctypeslib.as_array(self.wdl, shape=(self.size, 1)).astype("float32")) / 2

        return white_feature_tensor, black_feature_tensor, stm_tensor, score_tensor, wdl_tensor


BatchPtr = ctypes.POINTER(Batch)


class BatchReader(ctypes.Structure):
    _fields_ = [
        ('file', ctypes.c_void_p),
        ('batchSize', ctypes.c_uint),
        ('epoch', ctypes.c_uint)
    ]


BatchReaderPtr = ctypes.POINTER(BatchReader)

lib.newBatchReader.restype = BatchReaderPtr
lib.deleteBatchReader.restype = None
lib.getBatch.restype = BatchPtr
lib.deleteBatch.restype = None

lib.newBatchReader.argtypes = [ctypes.c_char_p, ctypes.c_uint]
lib.deleteBatchReader.argtypes = [BatchReaderPtr]
lib.getBatch.argtypes = [BatchReaderPtr]
lib.deleteBatch.argtypes = [BatchPtr]

new_batch_reader = lib.newBatchReader
delete_batch_reader = lib.deleteBatchReader
get_batch = lib.getBatch
delete_batch = lib.deleteBatch


class BatchProvider:
    def __init__(self, filename, batch_size, max_epoch):
        logging.info(f"Creating BatchReader object...")
        self.reader = new_batch_reader(filename.encode("utf8"), ctypes.c_uint(batch_size))

        if self.reader is None:
            logging.error("Unable to create BatchReader!")
            print("Unable to create BatchReader! Stopping...")
            exit(1)
        else:
            logging.info(f"Successfully created BatchReader.")

        self.batch = None
        self.max_epoch = max_epoch

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch is not None:
            delete_batch(self.batch)

        self.batch = get_batch(self.reader).contents

        if self.max_epoch < self.reader.contents.epoch:
            raise StopIteration

        return self.batch

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.batch is not None:
            delete_batch(self.batch)
        delete_batch_reader(self.reader)


