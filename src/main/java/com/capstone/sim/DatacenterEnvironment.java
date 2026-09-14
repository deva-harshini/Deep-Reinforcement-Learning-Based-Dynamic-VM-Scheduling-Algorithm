package com.capstone.sim;

import ch.qos.logback.classic.Level;
import org.cloudbus.cloudsim.brokers.DatacenterBroker;
import org.cloudbus.cloudsim.brokers.DatacenterBrokerSimple;
import org.cloudbus.cloudsim.cloudlets.Cloudlet;
import org.cloudbus.cloudsim.cloudlets.CloudletSimple;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.datacenters.Datacenter;
import org.cloudbus.cloudsim.datacenters.DatacenterSimple;
import org.cloudbus.cloudsim.hosts.Host;
import org.cloudbus.cloudsim.hosts.HostSimple;
import org.cloudbus.cloudsim.power.models.PowerModelHostSimple;
import org.cloudbus.cloudsim.resources.Pe;
import org.cloudbus.cloudsim.resources.PeSimple;
import org.cloudbus.cloudsim.schedulers.cloudlet.CloudletSchedulerTimeShared;
import org.cloudbus.cloudsim.schedulers.vm.VmSchedulerTimeShared;
import org.cloudbus.cloudsim.utilizationmodels.UtilizationModelDynamic;
import org.cloudbus.cloudsim.utilizationmodels.UtilizationModelFull;
import org.cloudbus.cloudsim.vms.Vm;
import org.cloudbus.cloudsim.vms.VmSimple;
import org.cloudsimplus.util.Log;

import java.util.ArrayList;
import java.util.List;

/**
 * CloudSimPlus Datacenter Environment simulating a 10-host, 20-VM power-aware cloud datacenter.
 * Provides reset() and step(action) methods for Gymnasium DRL interaction.
 */
public class DatacenterEnvironment {

    public static final int NUM_HOSTS = 10;
    public static final int NUM_VMS = 20;
    public static final int OBS_DIM = 20; // 10 CPU + 10 RAM utilizations
    public static final double STEP_DURATION_SEC = 10.0;
    public static final int MAX_STEPS = 500;

    // Reward function weights: R = -(alpha * Power + beta * SLAViolations + gamma * Migrations)
    private static final double ALPHA_POWER = 0.005;
    private static final double BETA_SLA = 2.0;
    private static final double GAMMA_MIGRATION = 0.5;
    private static final double SLA_CPU_THRESHOLD = 0.90;

    private CloudSim simulation;
    private Datacenter datacenter;
    private DatacenterBroker broker;
    private List<Host> hostList;
    private List<Vm> vmList;
    private List<Cloudlet> cloudletList;

    private int currentStep;
    private double cumulativeEnergyJoules;

    public DatacenterEnvironment() {
        Log.setLevel(Level.ERROR);
        initializeSimulation();
    }

    private void initializeSimulation() {
        this.simulation = new CloudSim();
        this.hostList = createHosts();
        this.datacenter = new DatacenterSimple(simulation, hostList);
        this.datacenter.setBandwidthPercentForMigration(0.5);

        this.broker = new DatacenterBrokerSimple(simulation);
        this.vmList = createVms();
        this.cloudletList = createCloudlets();

        this.broker.submitVmList(vmList);
        this.broker.submitCloudletList(cloudletList);

        this.currentStep = 0;
        this.cumulativeEnergyJoules = 0.0;

        // Start simulation entities and warm up
        this.simulation.startSync();
        this.simulation.runFor(1.0);
    }

    /**
     * Creates 10 power-aware hosts with heterogeneous specifications.
     * Hosts 0-4: Quad-Core, 10,000 MIPS total, 16 GB RAM (Max 250W, Static 100W)
     * Hosts 5-9: Dual-Core, 6,000 MIPS total, 8 GB RAM (Max 180W, Static 70W)
     */
    private List<Host> createHosts() {
        List<Host> list = new ArrayList<>(NUM_HOSTS);

        for (int i = 0; i < NUM_HOSTS; i++) {
            List<Pe> peList = new ArrayList<>();
            long ramMb;
            long bwMbps = 100_000;      // 100 Gbps
            long storageMb = 10_000_000; // 10 TB
            double maxPower;
            double staticPower;

            if (i < 5) {
                // Type A: High capacity quad-core host
                for (int p = 0; p < 4; p++) {
                    peList.add(new PeSimple(2500));
                }
                ramMb = 16384; // 16 GB
                maxPower = 250.0;
                staticPower = 100.0;
            } else {
                // Type B: Energy-efficient dual-core host
                for (int p = 0; p < 2; p++) {
                    peList.add(new PeSimple(3000));
                }
                ramMb = 8192; // 8 GB
                maxPower = 180.0;
                staticPower = 70.0;
            }

            Host host = new HostSimple(ramMb, bwMbps, storageMb, peList);
            host.setVmScheduler(new VmSchedulerTimeShared());
            host.setPowerModel(new PowerModelHostSimple(maxPower, staticPower));
            host.setId(i);
            list.add(host);
        }

        return list;
    }

    /**
     * Creates 20 VMs with diverse resource requirements.
     */
    private List<Vm> createVms() {
        List<Vm> list = new ArrayList<>(NUM_VMS);

        for (int i = 0; i < NUM_VMS; i++) {
            int pes;
            long mips;
            long ramMb;
            long bwMbps = 1000;
            long sizeMb = 10000;

            if (i < 8) {
                // Small VM
                pes = 1;
                mips = 1000;
                ramMb = 2048;
            } else if (i < 16) {
                // Medium VM
                pes = 2;
                mips = 1200;
                ramMb = 4096;
            } else {
                // Large VM
                pes = 2;
                mips = 2000;
                ramMb = 4096;
            }

            Vm vm = new VmSimple(i, mips, pes);
            vm.setRam(ramMb).setBw(bwMbps).setSize(sizeMb);
            vm.setCloudletScheduler(new CloudletSchedulerTimeShared());
            list.add(vm);
        }

        return list;
    }

    /**
     * Creates dynamic workload cloudlets with time-varying CPU demands.
     */
    private List<Cloudlet> createCloudlets() {
        List<Cloudlet> list = new ArrayList<>(NUM_VMS);
        long cloudletLength = 1_000_000_000L; // Long running

        for (int i = 0; i < NUM_VMS; i++) {
            Vm vm = vmList.get(i);
            Cloudlet cloudlet = new CloudletSimple(cloudletLength, (int) vm.getNumberOfPes());
            cloudlet.setVm(vm);

            // Dynamic utilization model with periodic workload variations
            final int vmIndex = i;
            UtilizationModelDynamic cpuModel = new UtilizationModelDynamic(0.4);
            cpuModel.setUtilizationUpdateFunction(model -> {
                double time = simulation.clock();
                double phase = (vmIndex * Math.PI / 10.0);
                double period = 100.0 + (vmIndex * 15.0);
                double base = 0.45;
                double amplitude = 0.35;
                double util = base + amplitude * Math.sin(2.0 * Math.PI * time / period + phase);
                return Math.max(0.1, Math.min(0.95, util));
            });

            cloudlet.setUtilizationModelCpu(cpuModel);
            cloudlet.setUtilizationModelRam(new UtilizationModelFull());
            list.add(cloudlet);
        }

        return list;
    }

    /**
     * Resets the datacenter simulation to initial state.
     * @return Initial observation vector of size 20 (10 CPU + 10 RAM utilizations).
     */
    public synchronized StepResult reset() {
        if (simulation != null && simulation.isRunning()) {
            simulation.abort();
        }
        initializeSimulation();
        double[] observation = getObservation();
        double totalPower = calculateTotalPowerWatts();
        return new StepResult(observation, 0.0, false, totalPower, 0, 0, currentStep, simulation.clock());
    }

    /**
     * Executes one action step in the datacenter environment.
     * @param action Target host IDs for each of the 20 VMs.
     * @return StepResult containing observation, reward, done flag, and metrics.
     */
    public synchronized StepResult step(int[] action) {
        currentStep++;
        int migrationsCount = 0;

        // 1. Process VM placement / migration actions
        if (action != null && action.length == NUM_VMS) {
            long[] availableRam = new long[NUM_HOSTS];
            double[] availableMips = new double[NUM_HOSTS];
            for (int h = 0; h < NUM_HOSTS; h++) {
                Host host = hostList.get(h);
                availableRam[h] = host.getRam().getAvailableResource();
                availableMips[h] = host.getVmScheduler().getTotalAvailableMips();
            }

            for (int i = 0; i < NUM_VMS; i++) {
                int targetHostId = action[i];
                if (targetHostId >= 0 && targetHostId < NUM_HOSTS) {
                    Vm vm = vmList.get(i);
                    Host currentHost = vm.getHost();
                    Host targetHost = hostList.get(targetHostId);

                    if (currentHost != null && !currentHost.equals(Host.NULL) && !currentHost.equals(targetHost)) {
                        long vmRam = vm.getRam().getCapacity();
                        double vmMips = vm.getTotalMipsCapacity();

                        if (availableRam[targetHostId] >= vmRam && availableMips[targetHostId] >= vmMips && targetHost.isSuitableForVm(vm)) {
                            datacenter.requestVmMigration(vm, targetHost);
                            availableRam[targetHostId] -= vmRam;
                            availableMips[targetHostId] -= vmMips;
                            availableRam[(int) currentHost.getId()] += vmRam;
                            availableMips[(int) currentHost.getId()] += vmMips;
                            migrationsCount++;
                        }
                    }
                }
            }
        }

        // 2. Advance CloudSim simulation clock
        simulation.runFor(STEP_DURATION_SEC);

        // 3. Compute real-time datacenter metrics
        double totalPowerWatts = calculateTotalPowerWatts();
        int slaViolations = calculateSlaViolations();
        cumulativeEnergyJoules += totalPowerWatts * STEP_DURATION_SEC;

        // 4. Calculate Scalar Reward Function
        // R_t = - ( alpha * PowerWatts + beta * SLAViolations + gamma * Migrations )
        double reward = -(ALPHA_POWER * totalPowerWatts + BETA_SLA * slaViolations + GAMMA_MIGRATION * migrationsCount);

        // 5. Get current observation vector
        double[] observation = getObservation();

        // 6. Termination condition
        boolean done = currentStep >= MAX_STEPS;

        return new StepResult(
                observation,
                reward,
                done,
                totalPowerWatts,
                slaViolations,
                migrationsCount,
                currentStep,
                simulation.clock()
        );
    }

    /**
     * Extracts normalized observation vector: [10 Host CPU utils, 10 Host RAM utils].
     */
    private double[] getObservation() {
        double[] obs = new double[OBS_DIM];
        for (int i = 0; i < NUM_HOSTS; i++) {
            Host host = hostList.get(i);
            // Host CPU utilization in [0.0, 1.0]
            double cpuUtil = host.getCpuPercentUtilization();
            obs[i] = Math.max(0.0, Math.min(1.0, cpuUtil));

            // Host RAM utilization in [0.0, 1.0]
            double ramUtil = host.getRam().getPercentUtilization();
            obs[NUM_HOSTS + i] = Math.max(0.0, Math.min(1.0, ramUtil));
        }
        return obs;
    }

    /**
     * Calculates total power draw (in Watts) across all 10 hosts.
     * Inactive/empty hosts are assumed powered off (0 Watts).
     */
    private double calculateTotalPowerWatts() {
        double totalPower = 0.0;
        for (Host host : hostList) {
            if (!host.getVmList().isEmpty()) {
                double cpuUtil = host.getCpuPercentUtilization();
                totalPower += host.getPowerModel().getPower(cpuUtil);
            }
        }
        return totalPower;
    }

    /**
     * Calculates SLA violations where host CPU demand exceeds SLA threshold.
     */
    private int calculateSlaViolations() {
        int violations = 0;
        for (Host host : hostList) {
            if (!host.getVmList().isEmpty()) {
                double cpuUtil = host.getCpuPercentUtilization();
                if (cpuUtil >= SLA_CPU_THRESHOLD) {
                    violations++;
                }
            }
        }
        return violations;
    }

    public synchronized void close() {
        if (simulation != null && simulation.isRunning()) {
            simulation.abort();
        }
    }

    /**
     * Data structure holding step results.
     */
    public record StepResult(
            double[] observation,
            double reward,
            boolean done,
            double powerWatts,
            int slaViolations,
            int migrations,
            int step,
            double simTime
    ) {}
}
