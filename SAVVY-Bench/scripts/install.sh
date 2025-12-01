# conda create -p /pscratch/sd/x/xiuliu/savvy python=3.12 -y
# conda activate /pscratch/sd/x/xiuliu/savvy

# git clone git@github.com:xxx
# cd SAVVY-Bench

export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12


# mount the benchmark and models
cp -r tasks/spatial_avqa third_party/lmms_eval/lmms_eval/tasks/


# install lmms-eval
cd third_party/lmms_eval
pip install -e .
cd ../../

# may take a long time
MAX_JOBS=4 pip install flash-attn==2.7.2.post1 --no-build-isolation
