#!/usr/bin/env python
import boto3

def get_ami_id():
    ec2 = boto3.client('ec2')
    response = ec2.describe_images(Owners=["self"],
                                   Filters=[{'Name': 'name',
                                             'Values': ['distributed_rl']}])
    return response['Images'][0]['ImageId']


def run_instance(image_id, instance_type, key_name, count=1):
    ec2 = boto3.client('ec2')
    response = ec2.run_instances(ImageId=image_id,
                                 InstanceType=instance_type,
                                 KeyName=key_name,
                                 MaxCount=count,
                                 MinCount=count)
    return [ins['InstanceId'] for ins in response['Instances']]

def wait_run(instance_ids):
    ec2 = boto3.client('ec2')
    waiter = ec2.get_waiter('instance_running')
    waiter.wait(InstanceIds=instance_ids)

def get_instance_ips(instance_ids):
    ec2 = boto3.client('ec2')
    instances = ec2.describe_instances(InstanceIds=instance_ids)
    return [ins['PublicIpAddress'] for ins in instances['Reservations'][0]['Instances']]

def run_instances_and_wait(config_file):
    import yaml
    config = yaml.load(open(config_file))
    image_id = get_ami_id()
    res_act = run_instance(image_id,
                           config['actors_instance_type'],
                           config['key_name'],
                           config['num_actors_instance'])
    res_learn = run_instance(image_id,
                             config['learner_instance_type'],
                             config['key_name'])
    res_all = res_act + res_learn
    print('Wait until instances running...')
    wait_run(res_act + res_learn)
    actors_ip = get_instance_ips(res_act)
    learner_ip = get_instance_ips(res_learn)
    print("Success in running all instances.")
    print("Learner's command: ",
          "fab -H %s -u ubuntu -i %s learner_run" % (learner_ip[0], config['ssh_key_file']))
    print("Actor's command: ",
          "fab -H %s -u ubuntu -i %s actor_run:num_proc=%d,leaner_host=%s" % (','.join(actors_ip),
                                                                              config['ssh_key_file'],
                                                                              config['num_actors_process_for_each_instance'],
                                                                              learner_ip[0]))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run instance script.')
    parser.add_argument('config_file', type=str, help='Config file name.')
    args = parser.parse_args()
    run_instances_and_wait(args.config_file)
