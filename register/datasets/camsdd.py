
from mmpretrain.datasets import BaseDataset
from mmpretrain.registry import DATASETS


# @DATASETS.register_module()
# class CamSDD(BaseDataset):
#     def __init__(self):
#         super().__init__(ann_file='')
#
# @DATASETS.register_module()
# class CamSDD32(BaseDataset):
#     def __init__(self):
#         super().__init__(ann_file='')

@DATASETS.register_module()
class CamSDD34(BaseDataset):
    def __init__(self):
        super().__init__(ann_file='')