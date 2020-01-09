from time import time

class OutTime:
    """
    Tool class to print execution time used by a piece of code.
    Usage:
    ```
        with OutTime():
            # code to execute...
    ```

    One can create custom timing class by inheriting from this class and overriding onEnter and/or onExit methods.
    """
    switchOff = False
    # set to True if you want to disable this tool.

    def __enter__(self):
        # record start time when scope is opened.
        if self.switchOff: return
        self.st = time()
        self.onEnter(self.st)
    
    def __exit__(self, type, value, traceback):
        # print execution time when scope is closed.
        if self.switchOff: return
        t = time() - self.st
        print("time passed: %5f s" % t)
        self.onExit(t)
    
    def onEnter(self, starttime):
        """
        Callback function executed after the start time is recorded.
        Args:
            starttime (float): the start time.
        """
        pass

    def onExit(self, time):
        """
        Callback function executed after the execution time is printed.
        Args:
            time (float): the execution time.

        NOTE: one can access the start time by ```self.st```
        """
        pass