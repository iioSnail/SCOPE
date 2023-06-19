#!/bin/bash

program_name="train.sh"

while true; do
   pid=$(ps -ef | grep "$program_name" | grep -v grep | awk '{print $2}')
   if [ -z "$pid" ]; then
         shutdown
   fi
   sleep 10
done