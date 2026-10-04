# -----------------------------------------------------------------------------
# CloudSim DRL Scheduler: Java CloudSimPlus Simulation Engine & ZeroMQ Server
# -----------------------------------------------------------------------------
FROM maven:3.9.6-eclipse-temurin-17 AS builder

WORKDIR /build

COPY pom.xml .
COPY src/ src/
RUN mvn clean package -DskipTests

FROM eclipse-temurin:17-jre-jammy

WORKDIR /app
COPY --from=builder /build/target/*.jar /app/app.jar
EXPOSE 5555

CMD ["java", "-cp", "app.jar:lib/*", "com.capstone.sim.CloudSimEnvServer"]
