import os
import sys
import torch
from detectron2.utils import comm
from detectron2.engine import hooks, HookBase
import logging

class PeriodicCheckpointerWithEval(HookBase):
    def __init__(self, eval_period, eval_function, checkpointer, checkpoint_period, max_to_keep=5):
        self.eval = hooks.EvalHook(eval_period, eval_function)
        self.checkpointer = hooks.PeriodicCheckpointer(checkpointer, checkpoint_period, max_to_keep=max_to_keep)
        self._logger = logging.getLogger("detectron2")
        best_model_path = checkpointer.save_dir + 'best_model_final.pth.pth'
        if os.path.isfile(best_model_path):
            best_model = torch.load(best_model_path, map_location=torch.device('cpu'))
            self.best_ap = best_model['AP50']
            del best_model
        else:
            self.best_ap = 0.0

    def before_train(self):
        self.max_iter = self.trainer.max_iter
        self.checkpointer.max_iter = self.trainer.max_iter

    def _do_eval(self):
        results = self.eval._func()
        comm.synchronize()
        return results

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self.eval._period > 0 and next_iter % self.eval._period == 0):
            results = self._do_eval()
            if comm.is_main_process():
                current_ap = results['bbox']['AP50']
                if current_ap > self.best_ap:
                    self.best_ap = current_ap
                    additional_state = {"iteration":self.trainer.iter, "AP50":self.best_ap}
                    self.checkpointer.checkpointer.save(
                    "best_model_final.pth", **additional_state
                    )
                    self._logger.info("Best AP: {}".format(self.best_ap))
                self._logger.info("Current AP: {}".format(current_ap))
        if comm.is_main_process():
            self.checkpointer.step(self.trainer.iter)
        comm.synchronize()

    def after_train(self):
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self.eval._func