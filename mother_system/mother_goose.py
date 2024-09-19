from multiprocessing import Pipe, Process
from g_manager import GManager
from d_manager import DManager

if __name__ == "__main__":
    g_pipe, d_pipe = Pipe(duplex=True)
    gm = GManager()
    dm = DManager()
    g_process = Process(target=gm.run, args=(g_pipe, ))
    d_process = Process(target=dm.run, args=(d_pipe, ))

    g_process.start()
    d_process.start()

    g_process.join()
    d_process.join()