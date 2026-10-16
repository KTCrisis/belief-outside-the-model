# Shared by inject/, probes/ and reset.sh. Run from anywhere.
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DC="docker compose -p stack-kafka-lab -f $LAB/compose.yaml"
BS="kafka:29092"
kbin() { $DC exec -T kafka /opt/kafka/bin/"$@"; }
knobs() { printf '%s\n' "$1" > "$LAB/state/knobs.json.tmp" && mv "$LAB/state/knobs.json.tmp" "$LAB/state/knobs.json"; }
total_lag() { kbin kafka-consumer-groups.sh --bootstrap-server $BS --describe --group lab 2>/dev/null | awk 'NR>2 && $6 ~ /^[0-9]+$/ {s+=$6} END {print s+0}'; }
