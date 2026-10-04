import sys
import time
import zmq
import json

def run_ipc_benchmark(num_messages=10000, endpoint="tcp://localhost:5555"):
    print("=" * 60)
    print(" Phase 0 IPC Benchmark: Python <-> Java (CloudSimPlus)")
    print("=" * 60)
    print(f"Target Server:          {endpoint}")
    print(f"Total Messages:         {num_messages:,}")
    print("-" * 60)

    context = zmq.Context()
    
    def create_socket():
        sock = context.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 2000) # 2 sec timeout
        sock.setsockopt(zmq.SNDTIMEO, 2000)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(endpoint)
        return sock

    socket = create_socket()
    print("[PYTHON] Connected to ZeroMQ server.")

    successful = 0
    dropped = 0
    latencies = []
    
    start_total_time = time.perf_counter()

    for i in range(1, num_messages + 1):
        payload = json.dumps({"step": i, "action": [0] * 20})
        t0 = time.perf_counter()
        
        try:
            socket.send_string(payload)
            reply = socket.recv_string()
            t1 = time.perf_counter()
            
            latencies.append((t1 - t0) * 1000.0)
            successful += 1
            
        except zmq.ZMQError as e:
            dropped += 1
            print(f"[ERROR] Message {i} failed: {e}")
            # Re-create socket to reset ZeroMQ REQ/REP state machine after failure
            socket.close()
            socket = create_socket()
            
            if dropped > 10:
                print("[ABORT] Too many dropped messages, stopping.")
                break

    total_time = time.perf_counter() - start_total_time
    
    print("=" * 60)
    print(" BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Time:             {total_time:.4f} s")
    print(f"Successful Messages:    {successful:,} / {num_messages:,}")
    print(f"Dropped Messages:       {dropped}")
    
    if successful > 0:
        throughput = successful / total_time
        print(f"Throughput Rate:        {throughput:,.2f} msgs/sec")
        print(f"Avg Round-Trip Latency: {sum(latencies)/len(latencies):.3f} ms")
    print("-" * 60)
    
    socket.close()
    context.term()

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    run_ipc_benchmark(count)