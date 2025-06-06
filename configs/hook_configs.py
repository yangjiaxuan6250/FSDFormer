# checkpoint saving
# checkpoint_config = dict(interval=1)
checkpoint_config = dict(type='ModelCheckpoint', indicator='loss', interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

# dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('simple_val', 1)]

# optimizer
optimizer = dict(type='AdamW', lr=3e-4)
optimizer_config = dict(grad_clip=None)
lr_config = None
# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=275)
