USE_GPU = True

DEVICE = 'cuda:0'
if not USE_GPU:
    DEVICE = 'cpu'

DATA_PATH = '/home/jlchen/sandbox/feature/dnn_paper'
BASE_PATH = '/home/jlchen/sandbox/'

ONBOARDING_DATA_PATH = '/home/jlchen/sandbox/onboarding/features/dnn_paper'
ONBOARDING_BASE_PATH = '/home/jlchen/sandbox/onboarding/'

WINDOW_SIZE = 64
SR = 16000
SEED = 2333


