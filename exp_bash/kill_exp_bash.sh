#!/bin/bash
ps aux | grep hzhaoaf | grep run_exp | grep -v 'grep' | awk '{print $2}'
echo 'kill all the above processes...'
ps aux | grep hzhaoaf | grep run_exp | grep -v 'grep' | awk '{print $2}' | xargs kill
echo 'finshed'
