"""A lightweight FIFO task queue to send several tasks to a background thread
and ensure they are processed in the right order.
"""

from Queue import Queue
from threading import Thread
import time

__all__ = ['JobQueue', 'jobqueue']


# JobQueue
# --------
class JobQueue(object):
    """Implements a queue containing jobs (Python methods of a base class
    specified in `cls`)."""
    def __init__(self, cls, *initargs, **initkwargs):
        self._queue = Queue()
        self._results = []
        # arguments of the task class constructor
        self.initargs, self.initkwargs = initargs, initkwargs
        # create the underlying task object
        self.task_class = cls
        self.task_obj = cls(*self.initargs, **self.initkwargs)
        # start the worker thread
        self.start()
        
    def start(self):
        """Start the worker thread."""
        self._thread = Thread(target=self._start)
        self._thread.daemon = True
        self._thread.start()
        
    def join(self):
        """Order to stop the queue as soon as all tasks have finished."""
        self._queue.put(None)
        self._thread.join()
        
    def _start(self):
        """Worker thread main function."""
        while True:
            r = self._queue.get()
            if r is not None:
                fun, args, kwargs = r
                self._results.append(fun(*args, **kwargs))
            else:
                break

    def _put(self, fun, *arg, **kwargs):
        """Put a function to process on the queue."""
        self._queue.put((fun, arg, kwargs))
        
    def __getattr__(self, name):
        return lambda *args, **kwargs: self._put(getattr(self.task_obj, name), *args, **kwargs)


def jobqueue(cls):
    class MyJobQueue(JobQueue):
        def __init__(self, *initargs, **initkwargs):
            super(MyJobQueue, self).__init__(cls, *initargs, **initkwargs)
    return MyJobQueue
        
        
    
    
if __name__ == '__main__':
    import sys
    
    # standard class with some long methods
    @jobqueue
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

