from time import time

class OutTime:
    switchOff = False
    def __enter__(self):
        if self.switchOff: return
        self.st = time()
        self.onEnter(self.st)
    
    def __exit__(self, type, value, traceback):
        if self.switchOff: return
        t = time() - self.st
        print("time passed: %5f s" % t)
        self.onExit(t)
    
    def onEnter(self, starttime):
        pass

    def onExit(self, time):
        pass