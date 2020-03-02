#!/usr/bin/env bash

total=500
session=5
count=0

while [ $count -lt $total ]
do
        ./basic.py $session
        count=$[$count+$session]
        echo "After $count episodes"
done
