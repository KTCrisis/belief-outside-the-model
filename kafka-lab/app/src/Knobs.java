import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;

/**
 * The fault knobs, re-read from /state/knobs.json at most once a second.
 * A missing file or key means the healthy default. Nothing here is logged:
 * the apps must not announce which fault is on.
 */
final class Knobs {
    private static final File FILE = new File("/state/knobs.json");
    private static final ObjectMapper JSON = new ObjectMapper();

    // producer
    volatile double rate = 300;        // messages per second
    volatile double skew = 0;          // share of messages sent to partition 0
    // consumers
    volatile double processMs = 4;     // CPU time spent per record
    volatile int gcLiveMb = 0;         // memory kept alive (old generation pressure)
    volatile int allocMbPerS = 0;      // garbage allocated per second
    volatile int rejoinEveryS = 0;     // leave and rejoin the group every n seconds
    volatile String crashMember = "";  // this consumer dies without leaving the group...
    volatile int crashEveryS = 0;      // ...after n seconds of uptime (docker restarts it)

    private long readAt = 0;

    Knobs refresh() {
        long now = System.currentTimeMillis();
        if (now - readAt < 1000) return this;
        readAt = now;
        JsonNode n;
        try {
            n = FILE.exists() ? JSON.readTree(FILE) : JSON.createObjectNode();
        } catch (Exception e) {
            return this;  // half-written file: keep the last values
        }
        rate = n.path("rate").asDouble(300);
        skew = n.path("skew").asDouble(0);
        processMs = n.path("process_ms").asDouble(4);
        gcLiveMb = n.path("gc_live_mb").asInt(0);
        allocMbPerS = n.path("alloc_mb_per_s").asInt(0);
        rejoinEveryS = n.path("rejoin_every_s").asInt(0);
        crashMember = n.path("crash_member").asText("");
        crashEveryS = n.path("crash_every_s").asInt(0);
        return this;
    }
}
