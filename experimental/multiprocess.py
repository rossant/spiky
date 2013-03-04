from multiprocessing import Process, Queue
from threading import Thread
import numpy as np
import time

FINISHED = '__END__'

def _run(taskcls, initargs, initkwargs, qin, qout):
    obj = taskcls(*initargs, **initkwargs)
    while True:
        p = qin.get()
        if p is None:
            # tell the client thread to shut down as all tasks have finished
            qout.put(FINISHED)
            break
        method, args, kwargs = p
        if hasattr(obj, method):
            # evaluate the method of the task object, and get the result
            result = getattr(obj, method)(*args, **kwargs)
            # send back the task arguments, and the result
            kwargs_back = kwargs.copy()
            kwargs_back.update(_result=result)
            qout.put((method, args, kwargs_back))


class JobQueueProcess(object):
    """Implements a queue containing jobs (Python methods of a base class
    specified in `cls`)."""
    def __init__(self, cls, *initargs, **initkwargs):
        self._queue = Queue()
        # self.done_callback = done_callback
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
        # start the client thread that processes the results sent back
        # from the worker process
        self._thread = Thread(target=self._retrieve)
        self._thread.daemon = True
        self._thread.start()
        
    def join(self):
        """Order to stop the queue as soon as all tasks have finished."""
        self._qin.put(None)
        self._process.join()

    def _put(self, fun, *args, **kwargs):
        """Put a function to process on the queue."""
        self._qin.put((fun, args, kwargs))
        
    def _retrieve(self):
        # call done_callback for each finished result
        while True:
            r = self._qout.get()
            if r == FINISHED:
                break
            # the method that has been called on the worked, with an additional
            # parameter _result in kwargs, containing the result of the task
            method, args, kwargs = r
            done_name = method + '_done'
            getattr(self.task_class, done_name)(*args, **kwargs)
        
    def __getattr__(self, name):
        if hasattr(self.task_class, name):
            return lambda *args, **kwargs: self._put(name, *args, **kwargs)


class MyTask(object):
    # task, called on the worker process
    def square(self, X):
        time.sleep(1)
        return X ** 2
        
    # callback task, called on the main process one the task has been done.
    # the signature must be the same as the corresponding task, with an 
    # additional keywoard argument: _result, with the task result.
    # It needs to be static. Typically it will fire a Qt event.
    @staticmethod
    def square_done(X, _result=None):
        if np.array_equal(_result, X**2):
            print("All good!")
        else:
            print("No good!")

        
if __name__ == '__main__':
    
    X = np.random.randn(100, 100)
    
    # create the task queue with the task class (must be available at top level)
    # and the callback function (called on the main process)
    task = JobQueueProcess(MyTask)
    
    # launch the execution of the task and returns immediately
    task.square(X)
    
    # wait for the tasks to finish, and shutdown everything
    task.join()
    
    
    