package com.capstone.sim;

import org.cloudbus.cloudsim.core.CloudSim;
import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;

public class IpcBenchmarkServer {
    public static void main(String[] args) {
        // 1. Verify CloudSim Engine Instantiation
        CloudSim simulation = new CloudSim();
        System.out.println("[JAVA] CloudSimPlus engine initialized successfully!");

        // 2. Open ZeroMQ Socket Server
        try (ZContext context = new ZContext()) {
            ZMQ.Socket socket = context.createSocket(SocketType.REP);
            socket.bind("tcp://*:5555");
            System.out.println("[JAVA] ZeroMQ IPC Server running on tcp://localhost:5555...");
            System.out.println("[JAVA] Waiting for Python DRL agent connection...");

            int stepCount = 0;
            while (!Thread.currentThread().isInterrupted()) {
                // Read incoming action from Python agent
                byte[] request = socket.recv(0);
                if (request == null) break;
                
                stepCount++;

                // Dummy state vector payload representing datacenter metrics
                String stateJson = String.format(
                    "{\"step\": %d, \"hosts_cpu\": [0.45, 0.82, 0.12, 0.95], \"reward\": -0.25, \"done\": false}", 
                    stepCount
                );

                // Send state vector back to Python agent
                socket.send(stateJson.getBytes(ZMQ.CHARSET), 0);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}