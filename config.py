Image_Width=384
Image_Height=96
Image_Channels=3

CHAR_LIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
NUM_CLASSES = len(CHAR_LIST) + 1

Bach_Size=32
Epochs=150
Learning_Rate=3e-4
Num_Workers=4

TRAIN_FOLDER = "train-data"
VAL_FOLDER   = "valid-data"
TEST_FOLDER  = "test-data"

CONFIDENCE_THRESHOLD = 0.2
ENTROPY_WEIGHT = 0.005
Seed = 42
Early_Stop_Patience = 10
Weight_Decay = 1e-4
