class Callback:
    def on_train_begin(self):
        pass

    def on_validation_end(self, logs, outputs, gt):
        pass

    def on_train_batch_end(self, loss):
        pass
