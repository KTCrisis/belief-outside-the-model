"""What a check returns, in words: for each test of each scenario, a short
description (what the check looks at, given to every agent) and phrasings of
a positive and of a negative result. A result is drawn from the scenario's
P(positive | cause); its text adds one irrelevant detail, so the reader has
to find the signal rather than match a keyword.
"""
import random

DESCRIPTIONS = {
    "kafka": {
        "rebalance_logs": "Read the consumer group coordinator logs for rebalances over the last 15 minutes.",
        "consumer_cpu_profile": "Profile CPU and per-record processing time inside the consumer instances.",
        "partition_lag_spread": "Compare the lag across the topic's partitions.",
        "broker_quota_metrics": "Read broker-side quota and throttle metrics for this client.",
        "jvm_gc_logs": "Read the consumers' JVM garbage collection logs.",
        "producer_rate_graph": "Look at the producers' message rate on the input topic.",
        "restart_consumer": "Restart one consumer instance and watch its lag for a few minutes.",
    },
    "internet": {
        "look_at_box_lights": "Ask the customer to describe the lights on the box.",
        "try_another_device": "Ask the customer to try the connection on another phone or laptop.",
        "restart_box": "Have the customer restart the box and test again.",
        "stand_next_to_box": "Ask the customer to test again standing right next to the box.",
        "check_cables": "Have the customer check that every cable is firmly plugged in.",
        "operator_status_page": "Have the customer open the operator's network status page on mobile data.",
        "ping_ip_vs_name": "Have the customer open a site by its IP address, then by its name.",
        "plug_ethernet": "Have the customer plug the computer into the box with an Ethernet cable.",
    },
}

TEXTS = {
    "kafka": {
        "rebalance_logs": (
            ["Coordinator log shows the group generation going from 41 to 57 in 12 minutes, members repeatedly leaving on session timeout.",
             "There are 14 'Preparing to rebalance group' entries in the window; partitions keep being revoked and reassigned."],
            ["Group generation stable at 41 for the whole window; no member left or joined.",
             "No rebalance entry in the coordinator log since yesterday; assignment unchanged."],
        ),
        "consumer_cpu_profile": (
            ["Processing time per record rose from 3 ms to 41 ms; the poll loop spends most of its time in the enrichment call.",
             "CPU pinned at 95% on all consumer pods, the handler is the hot path in the flame graph."],
            ["Per-record processing time is flat around 3 ms; CPU below 30% on all pods.",
             "The flame graph is dominated by network wait, handler time unchanged from last week."],
        ),
        "partition_lag_spread": (
            ["Partitions 3 and 7 hold 90% of the lag; the other ten are near zero.",
             "Lag is extremely uneven: one partition at 2.1M messages, the rest below 5k."],
            ["Lag is spread evenly, every partition between 180k and 210k.",
             "All twelve partitions show similar lag, no hot partition."],
        ),
        "broker_quota_metrics": (
            ["fetch-throttle-time-avg is 1800 ms for this client id, quota violations logged every minute.",
             "Brokers report the consumer's byte rate pinned exactly at its configured quota."],
            ["Throttle time is 0 for this client; no quota violation recorded.",
             "Byte rate well under quota, no throttling metric above zero."],
        ),
        "jvm_gc_logs": (
            ["Full GC pauses of 6 to 9 seconds every couple of minutes; old generation stays near 98% after collection.",
             "GC log shows stop-the-world pauses longer than the session timeout."],
            ["Young collections only, pauses under 40 ms; heap usage healthy.",
             "No full GC in the window, longest pause 25 ms."],
        ),
        "producer_rate_graph": (
            ["Input rate jumped from 2k to 19k msg/s at 09:12 and stayed there, right when the lag started.",
             "Producers are sending about eight times the usual rate since a batch job started."],
            ["Input rate flat at the usual 2k msg/s, same shape as the last seven days.",
             "No change in producer throughput around the start of the lag."],
        ),
        "restart_consumer": (
            ["After the restart the instance's lag dropped for a few minutes, then started climbing again.",
             "The restarted instance caught up briefly; the improvement faded within ten minutes."],
            ["Restarting the instance changed nothing; its lag kept growing at the same slope.",
             "No visible effect of the restart on lag or throughput."],
        ),
    },
    "internet": {
        "look_at_box_lights": (
            ["The customer says the internet light is red and blinking.",
             "The customer reports that one light that is usually white is now orange, or off."],
            ["The customer says all the lights look normal, white and steady.",
             "Lights as usual according to the customer: nothing red, nothing blinking."],
        ),
        "try_another_device": (
            ["The customer's phone on wifi does not load anything either.",
             "Tried on a second laptop: same problem, nothing loads."],
            ["On the phone everything works fine, only the laptop is stuck.",
             "Another device connects and browses normally."],
        ),
        "restart_box": (
            ["After the box rebooted, pages load again.",
             "The restart fixed it, the connection is back."],
            ["The box rebooted but nothing changed, still no connection.",
             "Same problem after the restart."],
        ),
        "stand_next_to_box": (
            ["Right next to the box, the connection works perfectly.",
             "Close to the box it loads fine; in the bedroom it fails."],
            ["Even standing next to the box, nothing loads.",
             "No difference near the box, still not working."],
        ),
        "check_cables": (
            ["The customer found the fiber cable half pulled out and pushed it back.",
             "One cable at the back of the box was loose."],
            ["All cables are firmly in place according to the customer.",
             "Every plug checked, nothing loose."],
        ),
        "operator_status_page": (
            ["The status page lists an ongoing incident in the customer's area.",
             "The operator reports a network outage for the customer's town."],
            ["The status page shows no incident in the area.",
             "Network status green for the customer's location."],
        ),
        "ping_ip_vs_name": (
            ["The site opens by its IP address but not by its name.",
             "Typing the numeric address works, the usual address does not."],
            ["Neither the IP address nor the name opens anything.",
             "Same result with the IP address and the name."],
        ),
        "plug_ethernet": (
            ["With the Ethernet cable the computer is online immediately.",
             "Wired, everything works."],
            ["Even with the Ethernet cable, still no connection.",
             "Plugged in by cable: no change."],
        ),
    },
}

NOISE = {
    "kafka": [
        "Unrelated: a dashboard panel shows a timezone warning.",
        "Note: the on-call channel mentions a planned certificate rotation next week.",
        "The schema registry answered normally during the check.",
        "Disk usage on the brokers is at 61%, unchanged.",
    ],
    "internet": [
        "The customer mentions the TV decoder was updated last week.",
        "The customer is calling from their mobile phone.",
        "The box was installed about two years ago.",
        "The customer adds that the weather is stormy today.",
    ],
}


def render(scenario: str, test: str, positive: bool, rng: random.Random) -> str:
    """The text a check returns for a given result: the signal, plus noise."""
    pos, neg = TEXTS[scenario][test]
    signal = rng.choice(pos if positive else neg)
    noise = rng.choice(NOISE[scenario])
    return f"{signal} {noise}" if rng.random() < 0.5 else f"{noise} {signal}"

# What a "positive" result means, given to the hybrid agent so the model can
# say how likely the signal is present in a text, and how likely it would be
# under each cause.
POSITIVE = {
    "kafka": {
        "rebalance_logs": "the group is rebalancing repeatedly",
        "consumer_cpu_profile": "processing in the consumers is slow or CPU-bound",
        "partition_lag_spread": "the lag is concentrated on a few partitions",
        "broker_quota_metrics": "the brokers are throttling this client",
        "jvm_gc_logs": "long garbage collection pauses",
        "producer_rate_graph": "the input rate rose sharply",
        "restart_consumer": "restarting an instance helped, at least for a while",
    },
    "internet": {
        "look_at_box_lights": "a light on the box is red, orange, blinking or off",
        "try_another_device": "another device fails too",
        "restart_box": "restarting the box fixed it",
        "stand_next_to_box": "it works right next to the box",
        "check_cables": "a cable was found loose",
        "operator_status_page": "the operator reports an outage in the area",
        "ping_ip_vs_name": "sites open by IP address but not by name",
        "plug_ethernet": "it works with an Ethernet cable",
    },
}

# What a negative result means, stated as plainly as POSITIVE: a closed-question
# reader (Jev) gets both sides as criteria, so a positive phrased as a double
# negative ("the phone does not load anything either") is not read as negative.
NEGATIVE = {
    "kafka": {
        "rebalance_logs": "the group is stable, no rebalance",
        "consumer_cpu_profile": "processing time and CPU in the consumers are normal",
        "partition_lag_spread": "the lag is spread evenly across partitions",
        "broker_quota_metrics": "no throttling of this client",
        "jvm_gc_logs": "garbage collection pauses are short and normal",
        "producer_rate_graph": "the input rate is unchanged",
        "restart_consumer": "restarting an instance changed nothing",
    },
    "internet": {
        "look_at_box_lights": "all lights on the box look normal",
        "try_another_device": "the other device connects and works normally (only the first one fails)",
        "restart_box": "still no connection after restarting the box",
        "stand_next_to_box": "it still fails even right next to the box",
        "check_cables": "every cable is firmly in place",
        "operator_status_page": "the operator reports no outage in the area",
        "ping_ip_vs_name": "IP address and name give the same result",
        "plug_ethernet": "it still fails with an Ethernet cable",
    },
}
