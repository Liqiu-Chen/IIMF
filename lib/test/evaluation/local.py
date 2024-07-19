from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.check_dir = '/models'
    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/he_zy/Chenlq/TBSI-main/data/got10k_lmdb'
    settings.got10k_path = '/he_zy/Chenlq/TBSI-main/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/he_zy/Chenlq/TBSI-main/data/itb'
    settings.lasot_extension_subset_path_path = '/he_zy/Chenlq/TBSI-main/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/he_zy/Chenlq/TBSI-main/data/lasot_lmdb'
    settings.lasot_path = '/he_zy/Chenlq/TBSI-main/data/lasot'
    settings.network_path = '/he_zy/Chenlq/TBSI-main-base-middlemodel-8/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/he_zy/Chenlq/TBSI-main/data/nfs'
    settings.otb_path = '/he_zy/Chenlq/TBSI-main/data/otb'
    settings.prj_dir = '/he_zy/Chenlq/TBSI-main-base-middlemodel-8'
    settings.result_plot_path = '/he_zy/Chenlq/TBSI-main-base-middlemodel-8/output/test/result_plots'
    settings.results_path = '/he_zy/Chenlq/TBSI-main-base-middlemodel-8/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/he_zy/Chenlq/TBSI-main-base-middlemodel-8/output'
    settings.segmentation_path = '/he_zy/Chenlq/TBSI-main-base-middlemodel-8/output/test/segmentation_results'
    settings.tc128_path = '/he_zy/Chenlq/TBSI-main/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/he_zy/Chenlq/TBSI-main/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/he_zy/Chenlq/TBSI-main/data/trackingnet'
    settings.uav_path = '/he_zy/Chenlq/TBSI-main/data/uav'
    settings.vot18_path = '/he_zy/Chenlq/TBSI-main/data/vot2018'
    settings.vot22_path = '/he_zy/Chenlq/TBSI-main/data/vot2022'
    settings.vot_path = '/he_zy/Chenlq/TBSI-main/data/VOT2019'
    settings.youtubevos_dir = ''
    settings.rgbt234_path = '/he_zy/Chenlq/TBSI-main/data/RGB-T234'
    settings.lasher_path = '/he_zy/Chenlq/TBSI-main/data/lasher/testingset'
    settings.rgbt210_path = '/he_zy/Chenlq/TBSI-main/data/RGBT210'
    return settings

