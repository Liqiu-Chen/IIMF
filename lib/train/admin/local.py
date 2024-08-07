class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/he_zy/Chenlq/TBSI-main-base-middlemodel-8'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/he_zy/Chenlq/TBSI-main-base-middlemodel-8/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/he_zy/Chenlq/TBSI-main-base-middlemodel-8/pretrained_networks'
        self.lasot_dir = '/he_zy/Chenlq/TBSI-main/data/lasot'
        self.got10k_dir = '/he_zy/Chenlq/TBSI-main/data/got10k/train'
        self.got10k_val_dir = '/he_zy/Chenlq/TBSI-main/data/got10k/val'
        self.lasot_lmdb_dir = '/he_zy/Chenlq/TBSI-main/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/he_zy/Chenlq/TBSI-main/data/got10k_lmdb'
        self.trackingnet_dir = '/he_zy/Chenlq/TBSI-main/data/trackingnet'
        self.trackingnet_lmdb_dir = '/he_zy/Chenlq/TBSI-main/data/trackingnet_lmdb'
        self.coco_dir = '/he_zy/Chenlq/TBSI-main/data/coco'
        self.coco_lmdb_dir = '/he_zy/Chenlq/TBSI-main/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/he_zy/Chenlq/TBSI-main/data/vid'
        self.imagenet_lmdb_dir = '/he_zy/Chenlq/TBSI-main/data/vid_lmdb'
        self.lasher_train_dir = '/he_zy/Chenlq/TBSI-main/data/lasher/trainingset/trainingset'
        self.lasher_test_dir = '/he_zy/Chenlq/TBSI-main/data/lasher/testingset/testingset'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
