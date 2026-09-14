package com.capstone.sim;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;

import java.util.HashMap;
import java.util.Map;

/**
 * ZeroMQ REP Server exposing CloudSimPlus DatacenterEnvironment to Python Gymnasium agent.
 * Listens on tcp://localhost:5555 for RESET, STEP, and CLOSE commands.
 */
public class CloudSimEnvServer {

    private static final int PORT = 5555;
    private static final Gson gson = new Gson();

    public static void main(String[] args) {
        System.out.println("=================================================");
        System.out.println(" CloudSimPlus Gymnasium Environment Server");
        System.out.println("=================================================");

        DatacenterEnvironment env = new DatacenterEnvironment();
        System.out.println("[JAVA] DatacenterEnvironment initialized successfully!");

        try (ZContext context = new ZContext()) {
            ZMQ.Socket socket = context.createSocket(SocketType.REP);
            socket.bind("tcp://*:" + PORT);
            System.out.printf("[JAVA] ZeroMQ Environment Server listening on tcp://*:%d%n", PORT);
            System.out.println("[JAVA] Ready for Gymnasium Python agent connections...");

            while (!Thread.currentThread().isInterrupted()) {
                byte[] requestBytes = socket.recv(0);
                if (requestBytes == null) {
                    break;
                }

                String requestStr = new String(requestBytes, ZMQ.CHARSET).trim();
                String responseJson;

                try {
                    JsonObject reqObj = JsonParser.parseString(requestStr).getAsJsonObject();
                    String command = reqObj.has("command") ? reqObj.get("command").getAsString().toUpperCase() : "STEP";

                    switch (command) {
                        case "RESET" -> {
                            DatacenterEnvironment.StepResult result = env.reset();
                            responseJson = formatStepResponse(result);
                        }
                        case "STEP" -> {
                            int[] action = new int[DatacenterEnvironment.NUM_VMS];
                            if (reqObj.has("action")) {
                                var actionArray = reqObj.getAsJsonArray("action");
                                for (int i = 0; i < actionArray.size() && i < action.length; i++) {
                                    action[i] = actionArray.get(i).getAsInt();
                                }
                            }
                            DatacenterEnvironment.StepResult result = env.step(action);
                            responseJson = formatStepResponse(result);
                        }
                        case "CLOSE" -> {
                            env.close();
                            Map<String, Object> closeRes = new HashMap<>();
                            closeRes.put("status", "closed");
                            responseJson = gson.toJson(closeRes);
                        }
                        default -> {
                            Map<String, Object> errRes = new HashMap<>();
                            errRes.put("error", "Unknown command: " + command);
                            responseJson = gson.toJson(errRes);
                        }
                    }
                } catch (Exception e) {
                    // Fallback for raw benchmark messages or parsing errors
                    DatacenterEnvironment.StepResult result = env.step(new int[DatacenterEnvironment.NUM_VMS]);
                    responseJson = formatStepResponse(result);
                }

                socket.send(responseJson.getBytes(ZMQ.CHARSET), 0);
            }
        } catch (Exception e) {
            System.err.println("[JAVA] Server exception: " + e.getMessage());
            e.printStackTrace();
        } finally {
            env.close();
            System.out.println("[JAVA] CloudSimEnvServer shut down.");
        }
    }

    private static String formatStepResponse(DatacenterEnvironment.StepResult result) {
        Map<String, Object> response = new HashMap<>();
        response.put("observation", result.observation());
        response.put("reward", result.reward());
        response.put("done", result.done());

        Map<String, Object> info = new HashMap<>();
        info.put("power_watts", result.powerWatts());
        info.put("sla_violations", result.slaViolations());
        info.put("migrations", result.migrations());
        info.put("step", result.step());
        info.put("sim_time", result.simTime());
        response.put("info", info);

        return gson.toJson(response);
    }
}
