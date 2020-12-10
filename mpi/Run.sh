#!/bin/sh

#read -p 'image_path: ' img_path
#read -p 'image_out: ' img_out

read -p 'program: ' program

img_paths=("4k.png" "720.jpg" "1024.jpg")
out_paths=("4k_480.png" "720_480.jpg" "1024_480.jpg")
size=("4k" "1080p" "720p")

path="../img/"
cmake . && make 

echo "size,threads,time" >> stats.txt
for  (( i = 0; i < 3; i++ ));
do
    for t in 2 4;
    do
        printf "${size[i]}," >> stats.txt
        mpirun -np $t  $program $path${img_paths[i]} $path${out_paths[i]}  >> stats.txt
    done
done 
