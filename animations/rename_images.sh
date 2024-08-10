#!/bin/bash
a=1
for i in $(ls fig_*.png | sort -V); do
  new=$(printf "fig_%03d.png" "$a") # fig_001.png, fig_002.png, ...
  mv -i -- "$i" "$new"
  let a=a+1
done

