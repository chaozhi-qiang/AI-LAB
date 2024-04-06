import mmpretrain
from mmpretrain.registry import DATASETS

# print(mmpretrain.list_models())


print(mmpretrain.CIFAR100)
print(mmpretrain.ImageNet21k)
print(mmpretrain.CustomDataset)
print(mmpretrain.CamSDD)
# print(mmpretrain.CamSDD32)

from mmpretrain import get_model
from mmpretrain import list_models
from mmpretrain import inference_model

# print(list_models(task='Image Classification', pattern='efficientnetv2-b0'))
# print(get_model('efficientnetv2-b0_3rdparty_in1k'))
# print(inference_model('efficientnetv2-b0_3rdparty_in1k', r'E:\openmm\mmproject\demo', show=True))


from mmpretrain.models import ImageClassifier

# from mmengine import Config
#
# cfg=Config.fromfile(r'./configs/custom_config/mobilenetv2_camsdd.py')
# print(cfg)




