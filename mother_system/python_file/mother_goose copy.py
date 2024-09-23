from multiprocessing import Pipe, Process, set_start_method
from g_manager import GManager
from d_manager_copy import DManager

if __name__ == "__main__":
    # spawn 시작 방식을 설정하여 cuda 관련 오류 방지
    set_start_method('spawn', force=True)
    
    g_pipe, d_pipe = Pipe(duplex=True)
    gm = GManager()
    dm = DManager()

    g_process = Process(target=gm.run, args=(g_pipe, ))
    d_process = Process(target=dm.run, args=(d_pipe, ))
    
    
    #d_process = Process(target=dm.__init__)
    g_process.start()
    d_process.start()

    g_process.join()
    d_process.join()