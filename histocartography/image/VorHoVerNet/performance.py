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
    DEFAULT_LOG_FUNCTION = lambda self, t: print("time passed: %5f s" % t)
    # set to True if you want to disable this tool.

    def __init__(self, log_function=None):
        if log_function is None:
            self.log_function = self.DEFAULT_LOG_FUNCTION
        else:
            self.log_function = log_function

    def __enter__(self):
        # record start time when scope is opened.
        if self.switchOff: return
        self.st = time()
        self.onEnter(self.st)
    
    def __exit__(self, type_, value, traceback):
        # print execution time when scope is closed.
        if self.switchOff: return
        t = time() - self.st
        try:
            self.log_function(t)
        except Exception as e:
            print(f"Custom log function threw an exception\n {type(e).__name__}: {e}")
            print("Roll back to default log function")
            self.DEFAULT_LOG_FUNCTION(t)
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

def test():
    i = 1
    with OutTime():
        i += 1

    with OutTime(lambda t: print(t)):
        i += 1

    with OutTime(lambda a, b: print(a, b)):
        i += 1

if __name__ == "__main__":
    test()