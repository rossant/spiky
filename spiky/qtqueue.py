import time
from spiky.queue import JobQueue
from galry import QtCore


__all__ = ['QtJobQueue', 'qtjobqueue']


# Qt JobQueue
# -----------
class QtJobQueue(JobQueue):
    """Job Queue supporting Qt signals and slots."""
    def start(self):
        jobself = self
        
        class JobQueueThread(QtCore.QThread):
            def run(self):
                jobself._start()
                
            def join(self):
                self.wait()
                
        self._thread = JobQueueThread()
        self._thread.start(QtCore.QThread.LowPriority)


def qtjobqueue(cls):
    class MyJobQueue(QtJobQueue):
        def __init__(self, *initargs, **initkwargs):
            super(MyJobQueue, self).__init__(cls, *initargs, **initkwargs)
    return MyJobQueue
    
    
if __name__ == '__main__':
    import sys
    
    # standard class with some long methods
    @qtjobqueue
    class MyTasks(object):
        def task1(self, arg):
            sys.stdout.flush()
            print "processing task1...", 
            time.sleep(1)
            print "ok"
            
        def task2(self, arg):
            sys.stdout.flush()
            print "processing task1...", 
            time.sleep(1)
            print "ok"
    
    # without the @jobqueue decorator, the task class would work normally
    # with the decorator, the tasks are rather sent asynchronously to a job
    # queue running in a background daemon thread, and processed serially.
    # It's up to the tasks to signal when they're done (in a Qt GUI, with
    # custom signals for instance).
    t = MyTasks()
    t.task1(3)
    t.task1(27)
    
    print "finished sending jobs"
    # without join, the process would terminate before the tasks would have
    # had the chance to finish. When using it in a GUI, this call is probably
    # not necessary.
    t.join()

