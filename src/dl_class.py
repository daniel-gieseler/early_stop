import numpy as np

class EarlyStopping:
    def __init__(self,
        final_step: int,
        log_step: float = 0.1,
        reference_curve: list[int, float] | None = None
    ):
        self.curve: list[float, float] = []
        self.log_step = log_step
        self.final_step = np.log(final_step, 10)
    
    def save_reference_curve(self):
        return

    
    def preprocess(self, step: int, loss: float) -> tuple[float, float] | None:
        if not self.curve:
            return None
        step_1 = np.log(step, 10)
        step_0, loss_0 = self.curve[-1]
        if step_1 - step_0 < self.log_step:
            return None
        loss_1 = np.log(loss, 10)
        loss_1 = min(loss_1, loss_0)
        return (step_1, loss_1)
    

    # extrapolate to a certain step
    # we have the end of the training by the learning rate scheduler!!
    # or the last step of the reference!


                

                




