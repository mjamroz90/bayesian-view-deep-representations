from datasets import celeb
from utils import logger


@logger.log
class Chairs3dDataset(celeb.CelebDataset):

    def __init__(self, chairs3d_dir, batch_size, scale_img):
        self.chairs3d_dir = chairs3d_dir
        super().__init__(chairs3d_dir, batch_size, None, 128, scale_img)
        self.logger.info("Chairs3D train/test dataset sizes = (%d, %d)" % (self.train_size, self.test_size))

    def settings(self):
        return {'ds_path': self.chairs3d_dir, 'image_size': self.out_size,
                'scale_img': self.scale_img, 'batch_size': self.batch_size}


def get_chairs3d_ds_from_train_config(train_config):
    if train_config['ds']['scale_img'] is True:
        scale_img_opt = '0_to_1'
    elif train_config['ds']['scale_img'] is not None:
        scale_img_opt = train_config['ds']['scale_img']
    else:
        scale_img_opt = None

    batch_size = train_config['ds']['batch_size'] if 'batch_size' in train_config['ds'] else 64
    chairs3d_ds = Chairs3dDataset(train_config['ds']['ds_path'], batch_size, scale_img_opt)
    return chairs3d_ds
