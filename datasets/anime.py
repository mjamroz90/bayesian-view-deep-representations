from datasets import celeb


class AnimeDataset(celeb.CelebDataset):

    def __init__(self, anime_dir, batch_size, scale_img):
        self.mini_imagenet_dir = anime_dir
        super().__init__(anime_dir, batch_size, None, 96, scale_img)
        self.logger.info("AnimeDataset train/test dataset sizes = (%d, %d)" % (self.train_size, self.test_size))


def get_anime_ds_from_train_config(train_config):
    if train_config['ds']['scale_img'] is True:
        scale_img_opt = '0_to_1'
    elif train_config['ds']['scale_img'] is not None:
        scale_img_opt = train_config['ds']['scale_img']
    else:
        scale_img_opt = None

    batch_size = train_config['ds']['batch_size'] if 'batch_size' in train_config['ds'] else 64
    imagenet_ds = AnimeDataset(train_config['ds']['ds_path'], batch_size, scale_img_opt)
    return imagenet_ds
