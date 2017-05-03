from time import time

from wavenet.utils import make_batch
from wavenet.models import Model, Generator

num_channels = 1
gpu_fraction = 1.0
num_classes = 2048

inputs, targets = make_batch('assets/voice.wav', num_classes)
num_time_samples = inputs.shape[1]

print inputs.shape, targets.shape, num_time_samples
model = Model(#num_time_samples=num_time_samples,
              num_channels=num_channels,
              gpu_fraction=gpu_fraction,
              num_classes=num_classes,
              prob_model_type='softmax')

tic = time()
model.train(inputs, targets)
toc = time()
print('Training took {} seconds.'.format(toc-tic))
