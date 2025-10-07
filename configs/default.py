class DefaultConfig:
    # Dataset settings
    img_size = 224
    num_workers = 4

    # Training settings
    batch_size = 32
    num_epochs = 10
    stability_window = 20  # Number of steps to measure layer stability
    freeze_threshold = 0.5  # MAD threshold for freezing layers

    # Optimization settings
    optimizer = 'AdamW'
    base_lr = 1e-5
    weight_decay = 0.01
    warmup_steps = 500

    # Model settings
    model_name = 'google/vit-base-patch16-224-in21k'  # ViT-B/16

    # HN-Freeze settings
    attention_threshold = 0.9  # Threshold for computing operational mode k

    # Misc
    seed = 42
    log_interval = 10
    eval_interval = 100
