#!/bin/sh
#
# decode_task{A,B,C}_run{1,2,3}.sh  task{A|B|C}_teamName_run{1|2|3}.csv

for((i=0;i<=39;i++))
do
{
    name="./chat_conv/"${i}"_unfluence.txt"
    if [ ! -f "$name" ]; then
        echo $name
        python divide_section.py --index=$i
    fi
    echo 'success'$i
}&     
done

python post_deal.py