from vocab import Vocabulary
import evaluation
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
RUN_PATH = "./models/model_dense_f30k.pth.tar"
DATA_PATH = "./data"
evaluation.evalrank(RUN_PATH, data_path=DATA_PATH, split="test")
