#! /usr/bin/env python
from nexus import job

def general_configs(machine):
    if machine=='taito':
        jobs = get_taito_configs()
    else:
        print 'Using taito as defaul machine'
        jobs = get_taito_configs()
    return jobs

def get_taito_configs():
    scf_presub = '''
    module purge
    module load gcc
    module load openmpi
    module load openblas
    module load hdf5-serial
    '''

    qmc_presub = '''
    module purge
    module load gcc/5.4.0
    module load mkl/11.3.2
    module load intelmpi/5.1.3
    module load hdf5-par/1.8.18
    module load fftw/3.3.6
    module load boost/1.63
    module load cmake/3.9.0
    '''

    qe='pw.x'
    p2q='pw2qmcpack.x'
    qmcpack = 'qmcpack_taito_cpu_comp_SoA'
    # 4 processes
    scf  = job(cores=4,minutes=5,user_env=False,presub=scf_presub,app=qe)
    conv  = job(cores=1,minutes=5,user_env=False,presub=scf_presub,app=p2q)
    vmc = job(cores=12,threads=2,minutes=5,user_env=False,presub=qmc_presub,app=qmcpack)
    qmc = job(cores=12,threads=2,minutes=15,user_env=False,presub=qmc_presub,app=qmcpack)

    jobs = {'scf' : scf, 'conv': conv, 'vmc': vmc, 'qmc': qmc}

    return jobs
