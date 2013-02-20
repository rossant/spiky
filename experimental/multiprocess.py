from multiprocessing import Process, Queue
import numpy as np


def _run(taskcls, initargs, initkwargs, qin, qout):
    obj = taskcls(*initargs, **initkwargs)
    while True:
        p = qin.get()
        if p is None:
            break
        method, args, kwargs = p
        if hasattr(obj, method):
            qout.put(getattr(obj, method)(*args, **kwargs))


class JobQueueProcess(object):
    """Implements a queue containing jobs (Python methods of a base class
    specified in `cls`)."""
    def __init__(self, cls, *initargs, **initkwargs):
        self._queue = Queue()
        self._results = []
        # If impatient, the queue will always process only the last tasks
        # and not the intermediary ones.
        self.impatient = initkwargs.pop('impatient', None)
        # arguments of the task class constructor
        self.initargs, self.initkwargs = initargs, initkwargs
        # create the underlying task object
        self.task_class = cls
        # start the worker thread
        self.start()
        
    def start(self):
        """Start the worker thread."""
        self._qin = Queue()
        self._qout = Queue()
        self._process = Process(target=_run, args=(self.task_class, 
            self.initargs, self.initkwargs, self._qin, self._qout,))
        self._process.start()
        
    def join(self):
        """Order to stop the queue as soon as all tasks have finished."""
        self._qin.put(None)
        self._process.join()

    def _put(self, fun, *args, **kwargs):
        """Put a function to process on the queue."""
        self._qin.put((fun, args, kwargs))
        return self._qout.get()
        
    def __getattr__(self, name):
        if hasattr(self.task_class, name):
            return lambda *args, **kwargs: self._put(name, *args, **kwargs)


class MyTask(object):
    def square(self, X):
        return X ** 2

if __name__ == '__main__':
    
    X = np.random.randn(100, 100)
    
    task = JobQueueProcess(MyTask)
    assert np.array_equal(task.square(X), X**2)
    
    task.join()
    
    