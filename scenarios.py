"""The diagnosis cases: causes with their prior, and tests with a cost in
minutes and P(positive | cause) for each cause, in the order of `causes`.

The numbers are hand-set to be plausible, not measured: they illustrate the
method, they are not statistics of real incidents.
"""

SCENARIOS = {
    # A Kafka consumer lag on an event platform (experiment 1).
    "kafka": {
        "miss": 60.0,  # minutes lost by acting on the wrong cause
        "causes": ["rebalancing", "slow_consumer", "partition_skew", "broker_throttling", "gc_pauses", "upstream_burst"],
        # what each cause means, given to every agent (experiment 2)
        "meaning": {
            "rebalancing": "the consumer group keeps rebalancing, so partitions are often unassigned",
            "slow_consumer": "the consumers' own processing per record became slow",
            "partition_skew": "a few partitions receive most of the traffic or are stuck",
            "broker_throttling": "the brokers throttle the consumers because of a quota",
            "gc_pauses": "long JVM garbage collection pauses freeze the consumers",
            "upstream_burst": "the producers suddenly send far more messages than usual",
        },
        "prior": [0.30, 0.20, 0.15, 0.10, 0.15, 0.10],
        "tests": [
            ("rebalance_logs", 2.0, [0.95, 0.05, 0.05, 0.05, 0.20, 0.05]),
            ("consumer_cpu_profile", 8.0, [0.10, 0.95, 0.10, 0.05, 0.40, 0.10]),
            ("partition_lag_spread", 3.0, [0.10, 0.10, 0.95, 0.05, 0.10, 0.25]),
            ("broker_quota_metrics", 4.0, [0.05, 0.05, 0.05, 0.95, 0.05, 0.05]),
            ("jvm_gc_logs", 5.0, [0.10, 0.25, 0.05, 0.05, 0.95, 0.05]),
            ("producer_rate_graph", 2.0, [0.10, 0.10, 0.25, 0.05, 0.05, 0.90]),
            ("restart_consumer", 1.0, [0.60, 0.40, 0.30, 0.30, 0.60, 0.30]),
        ],
    },
    # "My internet does not work": what a support assistant asks a customer
    # to check. Costs are the customer's minutes; a wrong diagnosis sends a
    # technician or a long hotline call for nothing.
    "internet": {
        "miss": 45.0,
        "causes": ["box_frozen", "weak_wifi", "isp_outage", "loose_cable", "device_issue", "dns"],
        "meaning": {
            "box_frozen": "the box itself is stuck and needs a restart; every device is affected",
            "weak_wifi": "the box works but the wifi signal is too weak where the customer is",
            "isp_outage": "the operator's network is down in the area; every device is affected",
            "loose_cable": "the line or fiber cable between the wall and the box is loose; every device is affected",
            "device_issue": "only the customer's computer is faulty (its wifi card or settings); other devices work",
            "dns": "the box's or operator's name resolution fails; every device is affected, sites still open by IP address",
        },
        "prior": [0.30, 0.20, 0.15, 0.10, 0.15, 0.10],
        "tests": [
            # positive = a box light is red, blinking or off
            ("look_at_box_lights", 1.0, [0.70, 0.10, 0.80, 0.90, 0.05, 0.10]),
            # positive = another phone or laptop fails too
            ("try_another_device", 2.0, [0.95, 0.50, 0.95, 0.95, 0.05, 0.90]),
            # positive = it works again after the reboot
            ("restart_box", 6.0, [0.90, 0.30, 0.05, 0.05, 0.10, 0.20]),
            # positive = it works right next to the box
            ("stand_next_to_box", 2.0, [0.05, 0.90, 0.02, 0.02, 0.10, 0.05]),
            # positive = a cable was found loose
            ("check_cables", 3.0, [0.05, 0.05, 0.05, 0.90, 0.05, 0.05]),
            # positive = the operator's status page (on 4G) lists an outage
            ("operator_status_page", 4.0, [0.05, 0.05, 0.85, 0.05, 0.05, 0.05]),
            # positive = sites open by IP address but not by name
            ("ping_ip_vs_name", 5.0, [0.05, 0.05, 0.05, 0.05, 0.05, 0.95]),
            # positive = it works with an Ethernet cable
            ("plug_ethernet", 8.0, [0.05, 0.95, 0.02, 0.02, 0.80, 0.05]),
        ],
    },
    # A patient with fever and cough, in winter. The currency is euros, not
    # minutes: each examination has a price, and a wrong diagnosis costs the
    # useless or missed treatment plus the second visit. The pedagogical
    # example of the theory document (volet I); numbers hand-set, plausible,
    # not clinical statistics.
    "clinic": {
        "miss": 200.0,  # euros lost by treating the wrong cause
        "causes": ["common_virus", "flu", "covid", "strep_throat", "pneumonia"],
        "meaning": {
            "common_virus": "a banal viral infection that heals alone",
            "flu": "influenza, with its own antiviral and its own course",
            "covid": "covid, with isolation and its own treatment",
            "strep_throat": "a bacterial throat infection that needs antibiotics",
            "pneumonia": "a bacterial lung infection that needs antibiotics and follow-up",
        },
        "prior": [0.35, 0.20, 0.15, 0.15, 0.15],
        "tests": [
            # positive = crackles heard in the lungs
            ("auscultation", 5.0, [0.05, 0.10, 0.10, 0.02, 0.80]),
            # positive = rapid strep test positive
            ("strep_rapid_test", 10.0, [0.05, 0.05, 0.05, 0.90, 0.05]),
            # positive = rapid antigen test positive
            ("covid_antigen_test", 15.0, [0.02, 0.05, 0.80, 0.02, 0.02]),
            # positive = rapid flu test positive
            ("flu_rapid_test", 20.0, [0.05, 0.85, 0.05, 0.02, 0.05]),
            # positive = CRP high in the blood
            ("blood_crp", 30.0, [0.10, 0.20, 0.25, 0.70, 0.90]),
            # positive = infiltrate on the chest X-ray
            ("chest_xray", 50.0, [0.02, 0.05, 0.10, 0.02, 0.90]),
        ],
    },
}
