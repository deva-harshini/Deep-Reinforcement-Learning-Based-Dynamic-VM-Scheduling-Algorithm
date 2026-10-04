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

public class DatacenterEnvironment {

    public static final int NUM_HOSTS = 10;
    public static final int NUM_VMS = 20;
    public static final int OBS_DIM = 40; // 10 CPU + 10 RAM + 10 BW + 10 Task Queue Load
    public static final double STEP_DURATION_SEC = 10.0;
    public static final int MAX_STEPS = 500;

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
    private double totalSlaThrottlingSeconds;
    private double totalExecutionSeconds;

    public DatacenterEnvironment() {
        Log.setLevel(Level.OFF);
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
        this.totalSlaThrottlingSeconds = 0.0;
        this.totalExecutionSeconds = 0.0;

        this.simulation.startSync();
        this.simulation.runFor(1.0);
    }

    private List<Host> createHosts() {
        List<Host> list = new ArrayList<>(NUM_HOSTS);
        for (int i = 0; i < NUM_HOSTS; i++) {
            List<Pe> peList = new ArrayList<>();
            long ramMb;
            long bwMbps = 100_000;
            long storageMb = 10_000_000;
            double maxPower, staticPower;

            if (i < 5) {
                for (int p = 0; p < 4; p++) peList.add(new PeSimple(2500));
                ramMb = 16384; maxPower = 250.0; staticPower = 100.0;
            } else {
                for (int p = 0; p < 2; p++) peList.add(new PeSimple(3000));
                ramMb = 8192; maxPower = 180.0; staticPower = 70.0;
            }

            Host host = new HostSimple(ramMb, bwMbps, storageMb, peList);
            host.setVmScheduler(new VmSchedulerTimeShared());
            host.setPowerModel(new PowerModelHostSimple(maxPower, staticPower));
            host.setId(i);
            list.add(host);
        }
        return list;
    }

    private List<Vm> createVms() {
        List<Vm> list = new ArrayList<>(NUM_VMS);
        for (int i = 0; i < NUM_VMS; i++) {
            int pes = (i < 8) ? 1 : 2;
            long mips = (i < 8) ? 1000 : (i < 16 ? 1200 : 2000);
            long ramMb = (i < 8) ? 2048 : 4096;

            Vm vm = new VmSimple(i, mips, pes);
            vm.setRam(ramMb).setBw(1000).setSize(10000);
            vm.setCloudletScheduler(new CloudletSchedulerTimeShared());
            list.add(vm);
        }
        return list;
    }

    private List<Cloudlet> createCloudlets() {
        List<Cloudlet> list = new ArrayList<>(NUM_VMS);
        for (int i = 0; i < NUM_VMS; i++) {
            Vm vm = vmList.get(i);
            Cloudlet cloudlet = new CloudletSimple(1_000_000_000L, (int) vm.getNumberOfPes());
            cloudlet.setVm(vm);

            final int vmIndex = i;
            UtilizationModelDynamic cpuModel = new UtilizationModelDynamic(0.4);
            cpuModel.setUtilizationUpdateFunction(model -> {
                double time = simulation.clock();
                double phase = (vmIndex * Math.PI / 10.0);
                double period = 100.0 + (vmIndex * 15.0);
                double baseWave = 0.45 + 0.35 * Math.sin(2.0 * Math.PI * time / period + phase);
                // Stochastic workload noise (+/- 10%) to model sudden cloud traffic surges and spikes
                double noise = java.util.concurrent.ThreadLocalRandom.current().nextDouble(-0.10, 0.10);
                double util = baseWave + noise;
                return Math.max(0.10, Math.min(0.98, util));
            });

            cloudlet.setUtilizationModelCpu(cpuModel);
            cloudlet.setUtilizationModelRam(new UtilizationModelFull());
            list.add(cloudlet);
        }
        return list;
    }

    public synchronized StepResult reset() {
        if (simulation != null && simulation.isRunning()) {
            simulation.abort();
        }
        initializeSimulation();
        return new StepResult(getObservation(), 0.0, false, calculateTotalPowerWatts(), 0, 0, 0, 0.0, 0.0, 0.0, currentStep, simulation.clock());
    }

    public synchronized StepResult step(int[] action) {
        currentStep++;
        int migrationsCount = 0;

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

        simulation.runFor(STEP_DURATION_SEC);

        double totalPowerWatts = calculateTotalPowerWatts();
        int slaViolations = calculateSlaViolations();
        int activeShutdowns = calculateActiveShutdowns();
        
        cumulativeEnergyJoules += totalPowerWatts * STEP_DURATION_SEC;
        totalExecutionSeconds += STEP_DURATION_SEC * NUM_HOSTS;
        if (slaViolations > 0) {
            totalSlaThrottlingSeconds += slaViolations * STEP_DURATION_SEC;
        }

        double slaPercent = (totalExecutionSeconds > 0) ? (totalSlaThrottlingSeconds / totalExecutionSeconds) * 100.0 : 0.0;
        double edp = cumulativeEnergyJoules * (simulation.clock() / currentStep); // Energy-Delay Product

        double reward = -(ALPHA_POWER * totalPowerWatts + BETA_SLA * slaViolations + GAMMA_MIGRATION * migrationsCount);
        boolean done = currentStep >= MAX_STEPS;

        return new StepResult(
                getObservation(), reward, done, totalPowerWatts, slaViolations,
                migrationsCount, activeShutdowns, edp, slaPercent, cumulativeEnergyJoules,
                currentStep, simulation.clock()
        );
    }

    /**
     * Extracts 40-dimensional state observation vector across all 10 hosts:
     * - [0..9]   : Host CPU Utilizations [0.0, 1.0]
     * - [10..19] : Host RAM Utilizations [0.0, 1.0]
     * - [20..29] : Host Network Bandwidth Utilizations [0.0, 1.0]
     * - [30..39] : Host Task Queue Load Ratios (VM count relative to total VMs) [0.0, 1.0]
     */
    private double[] getObservation() {
        double[] obs = new double[OBS_DIM];
        for (int i = 0; i < NUM_HOSTS; i++) {
            Host host = hostList.get(i);
            // 1. Host CPU Utilization
            obs[i] = Math.max(0.0, Math.min(1.0, host.getCpuPercentUtilization()));
            // 2. Host RAM Utilization
            obs[10 + i] = Math.max(0.0, Math.min(1.0, host.getRam().getPercentUtilization()));
            // 3. Host Network Bandwidth Utilization
            obs[20 + i] = Math.max(0.0, Math.min(1.0, host.getBw().getPercentUtilization()));
            // 4. Host Task Queue Load Ratio
            obs[30 + i] = Math.max(0.0, Math.min(1.0, (double) host.getVmList().size() / NUM_VMS));
        }
        return obs;
    }

    private double calculateTotalPowerWatts() {
        double totalPower = 0.0;
        for (Host host : hostList) {
            if (!host.getVmList().isEmpty()) {
                totalPower += host.getPowerModel().getPower(host.getCpuPercentUtilization());
            }
        }
        return totalPower;
    }

    private int calculateSlaViolations() {
        int violations = 0;
        for (Host host : hostList) {
            if (!host.getVmList().isEmpty() && host.getCpuPercentUtilization() >= SLA_CPU_THRESHOLD) {
                violations++;
            }
        }
        return violations;
    }

    private int calculateActiveShutdowns() {
        int shutdownCount = 0;
        for (Host host : hostList) {
            if (host.getVmList().isEmpty()) {
                shutdownCount++;
            }
        }
        return shutdownCount;
    }

    public synchronized void close() {
        if (simulation != null && simulation.isRunning()) {
            simulation.abort();
        }
    }

    public record StepResult(
            double[] observation, double reward, boolean done, double powerWatts,
            int slaViolations, int migrations, int activeShutdowns, double edp,
            double slaPercent, double totalEnergyJoules, int step, double simTime
    ) {}
}