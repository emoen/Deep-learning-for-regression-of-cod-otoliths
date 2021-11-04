ocker build --no-cache -t test_cupy .
docker run --rm -it -d --gpus device=1 --group-add 5003 --name run_cupy -v /scratch/disk2/Otoliths/endrem/deep:/gpfs/gpfs0/deep -p 88
88:8888 test_cupy
docker exec  -it run_cupy bash
