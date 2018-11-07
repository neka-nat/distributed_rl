#!/bin/bash
aws ec2 run-instances --image-id $(./get_ami_id.sh) --count 1 --instance-type p2.xlarge --key-name key
aws ec2 run-instances --image-id $(./get_ami_id.sh) --count $1 --instance-type t2.2xlarge --key-name key
