import ml_collections


def get_flaUnet_swin_B_config():
    config = ml_collections.ConfigDict()
    config.img_size = 224
    config.patch_size = 4
    config.in_chans = 3
    config.embed_dim = 128
    # config.depths = [2, 2, 18, 2]
    # config.depths = [2, 2, 18]
    config.depths = [2, 2]
    config.drop_rate = 0.0
    config.drop_path_rate = 0.5
    config.attn_drop_rate = 0.0
    config.num_heads = [4, 8, 16, 32]
    config.window_size = 56
    config.mlp_ratio = 4.
    config.qkv_bias = True
    config.qk_scale = None
    config.pretrain_ckpt = './model/flatten_swinb_pretrain_weight/ckpt_swin_b.pth'
    return config


def get_swinUnet_tiny_config():
    config = ml_collections.ConfigDict()
    # Model type
    config.MODEL = ml_collections.ConfigDict()
    config.MODEL.TYPE = 'swin'
    # Model name
    config.MODEL.NAME = 'swin_tiny_patch4_window7_224'
    # Checkpoint to resume, could be overwritten by command line argument
    config.MODEL.PRETRAIN_CKPT = './model/swin_transformer_pretrain_weight/swin_tiny_patch4_window7_224.pth'
    config.MODEL.RESUME = ''
    # Number of classes, overwritten in data preparation
    config.MODEL.NUM_CLASSES = 1000
    # Dropout rate
    config.MODEL.DROP_RATE = 0.0
    # Drop path rate
    config.MODEL.DROP_PATH_RATE = 0.1
    # Label Smoothing
    config.MODEL.LABEL_SMOOTHING = 0.1

    # Swin Transformer parameters
    config.MODEL.SWIN = ml_collections.ConfigDict()
    config.MODEL.SWIN.PATCH_SIZE = 4
    config.MODEL.SWIN.IN_CHANS = 3
    config.MODEL.SWIN.EMBED_DIM = 96
    config.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    config.MODEL.SWIN.DECODER_DEPTHS = [2, 2, 6, 2]
    config.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    config.MODEL.SWIN.WINDOW_SIZE = 7
    config.MODEL.SWIN.MLP_RATIO = 4.
    config.MODEL.SWIN.QKV_BIAS = True
    config.MODEL.SWIN.QK_SCALE = None
    config.MODEL.SWIN.APE = False
    config.MODEL.SWIN.PATCH_NORM = True
    config.MODEL.SWIN.FINAL_UPSAMPLE = "expand_first"

    config.TRAIN = ml_collections.ConfigDict()
    config.TRAIN.USE_CHECKPOINT = False
    return config


def get_gcvit_small_config():
    """return the GCViT_small configuration"""
    config = ml_collections.ConfigDict()
    # config.depths = [3, 4, 19, 5]
    config.depths = [3, 4]
    # config.num_heads = [3, 6, 12, 24]
    config.num_heads = [3, 6]
    # config.window_size = [7, 7, 14, 7]
    config.window_size = [7, 7]
    config.dim = 96
    config.mlp_ratio = 2
    config.drop_path_rate = 0.3
    config.layer_scale = 1e-5
    config.pretrain_ckpt = './model/gc_vit_pretrain_weight/gcvit_1k_small.pth.tar'

    config.vit_pretrain_ckpt = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    return config


def get_gcvit_small2_config():
    """return the GCViT_small2 configuration"""
    config = ml_collections.ConfigDict()
    # config.depths = [3, 4, 23, 5]
    config.depths = [3, 4, 23]
    config.num_heads = [3, 6, 12, 24]
    config.window_size = [7, 7, 14, 7]
    config.dim = 96
    config.mlp_ratio = 3
    config.drop_path_rate = 0.35
    config.layer_scale = 1e-5
    config.pretrain_ckpt = './model/gc_vit_pretrain_weight/gcvit_1k_small2.pth.tar'

    config.vit_pretrain_ckpt = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    return config


def get_gcvit_base_config():
    """return the GCVIT_base configuration"""
    config = ml_collections.ConfigDict()
    # config.depths = [3, 4, 19, 5]
    config.depths = [3, 4]
    # config.num_heads = [4, 8, 16, 32]
    config.num_heads = [4, 8]
    # config.window_size = [7, 7, 14, 7]
    config.window_size = [7, 7]
    config.dim = 128
    config.mlp_ratio = 2
    config.drop_path_rate = 0.5
    config.layer_scale = 1e-5
    config.pretrain_ckpt = './model/gc_vit_pretrain_weight/gcvit_1k_base.pth.tar'

    config.vit_pretrain_ckpt = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    return config
    return config


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    # custom
    config.classifier = 'seg'
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_r50_l16_config():
    """Returns the Resnet50 + ViT-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.resnet_pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    return config
