from time import time

from wavenet.utils import make_batch
from wavenet.models import Model, Generator

inputs, targets = make_batch('assets/voice.wav')
num_time_samples = inputs.shape[1]
num_channels = 1
gpu_fraction = 1.0


model = Model(num_time_samples=num_time_samples,
              num_channels=num_channels,
              gpu_fraction=gpu_fraction,
              prob_model_type='sdp')

tic = time()
model.train(inputs, targets)
toc = time()
print('Training took {} seconds.'.format(toc-tic))
