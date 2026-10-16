#!/usr/bin/env bash
# CPU use of each consumer over 5 s, and where its main thread is.
source "$(dirname "$0")/../lib.sh"
for c in consumer-1 consumer-2 consumer-3; do
  cpu=$($DC exec -T $c sh -c 'a=$(awk "{print \$14+\$15}" /proc/1/stat); sleep 5; b=$(awk "{print \$14+\$15}" /proc/1/stat); echo $(( (b-a) * 100 / 500 ))' 2>/dev/null)
  echo "$ $c: cpu ${cpu}% over 5 s; jcmd Thread.print (main)"
  $DC exec -T $c /opt/java/openjdk/bin/jcmd 1 Thread.print 2>/dev/null | awk '/^"main"/{f=1} f&&/^$/{exit} f' | head -8
done
