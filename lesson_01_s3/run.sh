#!/usr/bin/env bash 
export AWS_PROFILE=sachin-personal

bucket=sachin-mnist-2 # must be unique, won't work for you
aws s3 mb --region us-east-1 s3://$bucket

wget  http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P /tmp/
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P /tmp/

mkdir -p /tmp/data/
mv /tmp/train-*.gz /tmp/data/

for file in $(ls -d /tmp/data/*)
do
aws s3 cp $file s3://$bucket/data/mnist/
done
