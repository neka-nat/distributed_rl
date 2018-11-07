#!/bin/bash
aws ec2 describe-images --owner self --filters 'Name=name,Values=distributed_rl' --query 'Images[].[ImageId]' --output text|sort|tail -n 1
