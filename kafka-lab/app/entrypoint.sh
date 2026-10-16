#!/bin/sh
# producer | consumer. The consumers run with a small heap and the serial
# collector, so memory pressure (inject/gc_pauses.sh) gives real
# stop-the-world pauses; every JVM writes its GC log under /logs.
set -e
role="$1"
name="${NAME:-$role}"
cp="/app/classes:/libs/*"
log4j="-Dlog4j2.configurationFile=/app/log4j2.yaml -Dlab.name=$name"
case "$role" in
  producer)
    exec java -Xmx256m $log4j -cp "$cp" LabProducer ;;
  consumer)
    exec java -Xms512m -Xmx512m -XX:+UseSerialGC \
      "-Xlog:gc:file=/logs/gc-$name.log:time,uptime:filecount=2,filesize=2m" \
      $log4j -cp "$cp" LabConsumer ;;
  *)
    echo "usage: producer | consumer" >&2; exit 2 ;;
esac
