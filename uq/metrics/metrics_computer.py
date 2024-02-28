class MetricsComputer:
    def __init__(self, module):
        self.module = module
        self.rc = self.module.rc

    def should_compute(self, stage):
        return (
            (stage == 'train' and self.rc.config.save_train_metrics)
            or (stage == 'val' and self.rc.config.save_val_metrics)
            or (stage == 'test' and self.rc.config.save_test_metrics)
            or (stage == 'calib')
        )

    def compute_monitor(self, monitor_value):
        if self.module.interleaved and self.module.current_epoch % 2 == 0:
            return {}
        return {
            self.module.monitor: monitor_value,
        }

    def compute(self, stage=None):
        if stage == 'train':
            return self.train_metrics()
        elif stage == 'val':
            return self.val_metrics()
        elif stage == 'calib':
            return self.val_metrics()
        elif stage == 'test':
            return self.test_metrics()
        else:
            raise RuntimeError('Invalid stage')
