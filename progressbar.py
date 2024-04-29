import sys
import time

### May not work on some IDEs (on Linux terminal works well) ###

class ProgressBar:
    """ Simple progressbar to track progress"""
    def __init__(self, total):
        """
        Initialize a ProgressBar instance with total steps required for completion.

        Parameters:
            total (int): The total number of iterations or steps the progress bar is tracking.
        """
        self.total = total
        self.start_time = time.time()
        self.bar_length = 40

    def update(self, current):
        """
        Update the progress bar to reflect the current status of the process.

        Parameters:
            current (int): The current iteration or position in the process.
        """
        elapsed_time = time.time() - self.start_time
        percent = (current / self.total)
        filled_length = int(self.bar_length * current // self.total)
        bar = '#' * filled_length + '-' * (self.bar_length - filled_length)
        if current < self.total:
            eta = elapsed_time / percent * (1 - percent)
        else:
            eta = 0
        elapsed_minutes = elapsed_time / 60
        eta_minutes = eta / 60
        print(f'\rProgress: |{bar}| {current}/{self.total} ({percent*100:.2f}%) Elapsed: {elapsed_minutes:.2f} min ETA: {eta_minutes:.2f} min', end='\r')

    def finish(self):
        """
        Mark the completion of the progress, finalizing the progress bar output.
        """
        print('\nCompleted!')
