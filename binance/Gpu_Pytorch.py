import pytorch_lightning as pl
import torch

print('pytorch version %s' % torch.__version__)
print('pytorch lightning version %s' % pl.__version__)
print('cuda version %s' % torch.version.cuda)
print('cudnn version %s' % torch.backends.cudnn.version())
print('gpu available %s' % torch.cuda.is_available())
print('gpu device %s' % torch.cuda.current_device())
print('gpu name %s' % torch.cuda.get_device_name())
