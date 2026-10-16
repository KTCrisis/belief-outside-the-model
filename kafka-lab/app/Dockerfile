# The producer and the consumers, built against the client jars of the very
# Kafka image the broker runs: nothing is downloaded.
FROM apache/kafka:4.0.0 AS kafka

FROM eclipse-temurin:21-jdk
COPY --from=kafka /opt/kafka/libs /libs
WORKDIR /app
COPY src/ src/
COPY log4j2.yaml entrypoint.sh ./
RUN javac -cp "/libs/*" -d classes src/*.java && chmod +x entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
