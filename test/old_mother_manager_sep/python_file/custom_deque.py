from collections import deque
import threading

class CircularQueue:
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)
        self.lock = threading.Lock()  # 멀티스레드 환경에서의 동기화

    def put(self, item):
        with self.lock:  # 큐에 접근할 때 잠금 처리
            self.queue.append(item)
            # print(f"Added: {item}, Queue: {list(self.queue)}")

    def get(self):
        with self.lock:  # 큐에서 데이터를 가져올 때 잠금 처리
            if len(self.queue) > 0:
                item = self.queue.popleft()
                # print(f"Removed: {item}, Queue: {list(self.queue)}")
                return item
            else:
                return None

    def size(self):
        with self.lock:
            return len(self.queue)