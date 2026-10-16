import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.serialization.StringSerializer;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Properties;
import java.util.Random;

/** Order events at a target rate; a share of them can be pinned to partition 0. */
public final class LabProducer {
    private static final Logger LOG = LogManager.getLogger("lab.producer");

    public static void main(String[] args) throws Exception {
        String topic = System.getenv().getOrDefault("TOPIC", "orders");
        Properties p = new Properties();
        p.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, System.getenv().getOrDefault("BOOTSTRAP", "kafka:29092"));
        p.put(ProducerConfig.CLIENT_ID_CONFIG, "order-service");
        p.put(ProducerConfig.ACKS_CONFIG, "1");
        p.put(ProducerConfig.LINGER_MS_CONFIG, "5");
        p.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        p.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());

        Knobs k = new Knobs();
        Random rng = new Random();
        String pad = "x".repeat(380);
        long sent = 0, windowStart = System.currentTimeMillis(), seq = 0;
        double credit = 0;
        long last = System.nanoTime();
        try (KafkaProducer<String, String> producer = new KafkaProducer<>(p)) {
            while (true) {
                k.refresh();
                long now = System.nanoTime();
                credit = Math.min(credit + k.rate * (now - last) / 1e9, k.rate);  // at most one second of backlog
                last = now;
                while (credit >= 1) {
                    credit -= 1;
                    String key = "customer-" + rng.nextInt(10_000);
                    String value = "{\"order\":" + (seq++) + ",\"customer\":\"" + key + "\",\"amount\":"
                            + rng.nextInt(50_000) / 100.0 + ",\"note\":\"" + pad + "\"}";
                    ProducerRecord<String, String> rec = rng.nextDouble() < k.skew
                            ? new ProducerRecord<>(topic, 0, key, value)
                            : new ProducerRecord<>(topic, key, value);
                    producer.send(rec);
                    sent++;
                }
                long wall = System.currentTimeMillis();
                if (wall - windowStart >= 10_000) {
                    LOG.info("sent {} msg/s", Math.round(sent * 1000.0 / (wall - windowStart)));
                    sent = 0;
                    windowStart = wall;
                }
                Thread.sleep(5);
            }
        }
    }
}
