#!/usr/bin/env python3
"""
Phase 0 IPC Benchmark Test: Java (CloudSimPlus/JeroMQ) <-> Python (ZeroMQ)
Validates communication throughput, latency, and ensures zero dropped messages.
"""

import json
import time
import zmq
import sys

def run_ipc_benchmark(num_messages: int = 10000, server_endpoint: str = "tcp://localhost:5555"):
    print("=" * 60)
    print(f" Phase 0 IPC Benchmark: Python <-> Java (CloudSimPlus)")
    print("=" * 60)
    print(f"Target Server: {server_endpoint}")
    print(f"Total Messages: {num_messages:,}")
    print("-" * 60)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
    socket.setsockopt(zmq.SNDTIMEO, 5000)

    try:
        socket.connect(server_endpoint)
        print("[PYTHON] Connected to ZeroMQ server.")
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        return False

    latencies_us = []
    dropped_messages = 0
    start_total = time.perf_counter()

    sample_action = json.dumps({"action": [1, 0, 3], "type": "STEP"})

    for i in range(1, num_messages + 1):
        t0 = time.perf_counter()
        try:
            socket.send_string(sample_action)
            reply = socket.recv_string()
            t1 = time.perf_counter()

            # Record round-trip latency in microseconds
            latency_us = (t1 - t0) * 1_000_000
            latencies_us.append(latency_us)

            # Sample verification of payload structure
            if i == 1 or i == num_messages or i % 2500 == 0:
                data = json.loads(reply)
                assert "step" in data, "Missing 'step' key in response"
                assert "hosts_cpu" in data, "Missing 'hosts_cpu' key in response"
                assert "reward" in data, "Missing 'reward' key in response"
                assert "done" in data, "Missing 'done' key in response"

        except Exception as e:
            print(f"[ERROR] Message {i} failed: {e}")
            dropped_messages += 1
            if dropped_messages > 10:
                print("[ABORT] Too many dropped messages, stopping.")
                break

    end_total = time.perf_counter()
    total_time_sec = end_total - start_total

    successful_messages = num_messages - dropped_messages
    throughput = successful_messages / total_time_sec if total_time_sec > 0 else 0
    avg_latency_ms = (sum(latencies_us) / len(latencies_us)) / 1000.0 if latencies_us else 0
    min_latency_ms = (min(latencies_us)) / 1000.0 if latencies_us else 0
    max_latency_ms = (max(latencies_us)) / 1000.0 if latencies_us else 0

    # Percentiles
    sorted_latencies = sorted(latencies_us)
    p50_ms = sorted_latencies[int(len(sorted_latencies) * 0.50)] / 1000.0 if sorted_latencies else 0
    p95_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)] / 1000.0 if sorted_latencies else 0
    p99_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)] / 1000.0 if sorted_latencies else 0

    print("=" * 60)
    print(" BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Time:             {total_time_sec:.4f} s")
    print(f"Successful Messages:    {successful_messages:,} / {num_messages:,}")
    print(f"Dropped Messages:       {dropped_messages}")
    print(f"Throughput Rate:        {throughput:,.2f} msgs/sec")
    print(f"Avg Round-Trip Latency: {avg_latency_ms:.3f} ms ({avg_latency_ms * 1000:.1f} µs)")
    print(f"Min / Max Latency:      {min_latency_ms:.3f} ms / {max_latency_ms:.3f} ms")
    print(f"50th Percentile (p50):  {p50_ms:.3f} ms")
    print(f"95th Percentile (p95):  {p95_ms:.3f} ms")
    print(f"99th Percentile (p99):  {p99_ms:.3f} ms")
    print("-" * 60)

    target_throughput = 1000.0
    passed = dropped_messages == 0 and throughput >= target_throughput
    if passed:
        print(f"✅ PHASE 0 BENCHMARK PASSED: Throughput {throughput:,.2f} msgs/sec > {target_throughput:,.0f} msgs/sec threshold!")
    else:
        print(f"❌ PHASE 0 BENCHMARK FAILED: Target was >{target_throughput:,.0f} msgs/sec with 0 drops.")

    print("=" * 60)

    socket.close()
    context.term()

    return passed

if __name__ == "__main__":
    count = 10000
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
    success = run_ipc_benchmark(num_messages=count)
    sys.exit(0 if success else 1)
