from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode

dataset_subtrain = MsDataset.load('COCO2017_Instance_Segmentation', split='subtrain', cache_dir='datasets/train', 
                                  download_mode=DownloadMode.FORCE_REDOWNLOAD)
dataset_val = MsDataset.load('COCO2017_Instance_Segmentation', split='validation/', cache_dir='datasets/eval', 
                             download_mode=DownloadMode.FORCE_REDOWNLOAD)