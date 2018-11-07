#!/bin/bash
aws ec2 describe-instances --filter "Name=instance-type,Values=$1" | jq '.Reservations [] .Instances [] .PublicIpAddress'
