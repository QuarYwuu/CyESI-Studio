"""CPU budgets and spawn-safe background jobs."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import ctypes
import multiprocessing as mp
import os
import queue

_MESSAGES=None
_LIMITER=None


def available_memory():
    if os.name=='nt':
        class Memory(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
                ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
        info=Memory();info.length=ctypes.sizeof(info)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(info)):return info.available
    try:return os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
    except (AttributeError,ValueError,OSError):return 4*1024**3


def worker_budget(requested=0,jobs=32):
    logical=os.cpu_count() or 1
    if requested:return max(1,min(requested,logical,jobs,32))
    cpu=max(1,min(8,logical//2-1))
    # Leave 1.5 GiB free; allow about 0.75 GiB per independently meshed solve.
    memory=max(1,int((available_memory()-1.5*1024**3)/(.75*1024**3)))
    return max(1,min(cpu,memory,jobs))


def emit(message):
    if _MESSAGES is not None:_MESSAGES.put(str(message))


def _initialize(messages):
    global _MESSAGES,_LIMITER
    _MESSAGES=messages
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[name]='1'
    try:
        from threadpoolctl import threadpool_limits
        _LIMITER=threadpool_limits(limits=1)
    except ImportError:pass


def process_jobs(function,jobs,workers,progress=lambda s:None):
    """Yield (input index, result) in completion order; propagate worker errors."""
    context=mp.get_context('spawn');messages=context.Queue()
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=_initialize,initargs=(messages,)) as pool:
            pending={pool.submit(function,job):i for i,job in enumerate(jobs)}
            while pending:
                done,_=wait(pending,timeout=.15,return_when=FIRST_COMPLETED)
                while True:
                    try:progress(messages.get_nowait())
                    except queue.Empty:break
                for future in done:
                    index=pending.pop(future)
                    yield index,future.result()
    finally:messages.close();messages.join_thread()
