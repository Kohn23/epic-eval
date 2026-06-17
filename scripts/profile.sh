# do not use source
ncu --set full --target-processes all -f -o epic_profile/test ./build/epic_driver -b tpccfull -d epic -w 1 -a 0.0 -r true -c 32 -e 5 -s 100000 -f true -m false -n 10000000 -x gpu