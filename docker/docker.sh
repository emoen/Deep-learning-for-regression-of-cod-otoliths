docker build --no-cache -t test_cupy2 .
docker run --rm -it -d --gpus device=0 --group-add 5003 --name run_cupy2 -v /scratch/disk2/Otoliths/endrem/deep:/gpfs/gpfs0/deep -p 7776:7776 test_cupy2
docker exec  -it run_cupy2 bash


