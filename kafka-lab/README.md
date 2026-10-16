# kafka-lab

A small Kafka platform that breaks on demand. It serves the research on
investigation agents with an explicit Bayesian belief (real faults, real
check outputs, known cause).

One broker (Kafka 4.0, KRaft), one producer (`order-service`, 300 orders/s on
`orders`, 6 partitions), a group `lab` of three Java consumers. The apps are
built from the client jars of the broker image itself, so nothing else is
downloaded.

```sh
docker compose -p stack-kafka-lab up -d
./inject/slow_consumer.sh    # a fault
./probes/partition_lag_spread.sh
./reset.sh                   # fresh topic and group, empty logs
```

## Faults (`inject/`)

| Script | What actually happens |
| --- | --- |
| `rebalancing.sh` | consumer-1 crash-loops: dies after 20 s without leaving the group, docker restarts it |
| `slow_consumer.sh` | per-record processing goes from 4 to 20 ms of CPU |
| `partition_skew.sh` | 98 % of the orders go to partition 0 |
| `broker_throttling.sh` | a default client quota caps fetches at 15 kB/s |
| `gc_pauses.sh` | 470 MB kept alive on 512 MB heaps, plus churn: long full GCs of the serial collector |
| `upstream_burst.sh` | the producer sends 3,000 orders/s instead of 300 |
| `clear.sh` | back to the healthy defaults |

`reset.sh` goes further, between episodes: it deletes the group and recreates
the topic, so no offset, lag or log line survives from one episode to the next.

Most faults are knobs in `state/knobs.json`, re-read every second by the apps;
the apps never log which knob is on. The quota goes through `kafka-configs`.

## Checks (`probes/`)

Each prints what an on-call engineer would read, raw.

| Script | Output |
| --- | --- |
| `rebalance_logs.sh` | group coordination lines from the consumers' logs, last 2 minutes |
| `partition_lag_spread.sh` | `kafka-consumer-groups --describe --group lab` |
| `producer_rate_graph.sh` | incoming msg/s per partition, from log-end offsets 10 s apart |
| `broker_quota_metrics.sh` | default client quota, and the consumers' fetch throttle time |
| `jvm_gc_logs.sh` | the last GC events of each consumer |
| `consumer_cpu_profile.sh` | CPU over 5 s and the main thread's stack (`jcmd Thread.print`) |
| `restart_consumer.sh` | restart one consumer and follow the total lag for 30 s |

Note: `reset.sh` restarts the apps, which join the group; wait about two
minutes after a reset before injecting if the coordination lines of the
start-up should not appear in `rebalance_logs.sh`.
