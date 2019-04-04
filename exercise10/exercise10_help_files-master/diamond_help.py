#! /usr/bin/env python

from nexus import settings,job,run_project
from nexus import generate_physical_system
from nexus import generate_pwscf
from nexus import generate_pw2qmcpack
from nexus import generate_qmcpack,vmc,dmc
from machine_configs import get_taito_configs
from optim_params import *

settings(
    pseudo_dir    = './pseudopotentials',
    status_only   = 0,
    generate_only = 1,
    sleep         = 3,
    machine       = 'taito'
    )

jobs = get_taito_configs()

dia16 = generate_physical_system(
    units  = 'A',
    axes   = [[ 1.785,  1.785,  0.   ],
              [ 0.   ,  1.785,  1.785],
              [ 1.785,  0.   ,  1.785]],
    elem   = ['C','C'],
    pos    = [[ 0.    ,  0.    ,  0.    ],
              [ 0.8925,  0.8925,  0.8925]],
    tiling = (2,2,2),
    kgrid  = (1,1,1),
    kshift = (0,0,0),
    C      = 4
    )
              
scf = generate_pwscf(
    identifier   = 'scf',
    path         = 'diamond/scf',
    job          = jobs['scf'],
    input_type   = 'generic',
    calculation  = 'scf',
    input_dft    = 'lda', 
    ecutwfc      = 200,   
    conv_thr     = 1e-8, 
    nosym        = True,
    wf_collect   = True,
    system       = dia16,
    pseudos      = ['C.BFD.upf'], 
    )

conv = generate_pw2qmcpack(
    identifier   = 'conv',
    path         = 'diamond/scf',
    job          = jobs['conv'],
    write_psir   = False,
    dependencies = (scf,'orbitals'),
    )

qmc = generate_qmcpack(
    identifier   = 'vmc',
    path         = 'diamond/vmc',
    job          = jobs['vmc'],
    input_type   = 'basic',
    system       = dia16,
    pseudos      = ['C.BFD.xml'],
    jastrows     = [],
    calculations = [
        vmc(
            walkers     =   1,
            warmupsteps =  20,
            blocks      = 200,
            steps       =  10,
            substeps    =   2,
            timestep    =  .4
            )
        ],
    dependencies = (conv,'orbitals'),
    )

optims = getOptims()
optJ2 = generate_qmcpack(
    path = 'diamond/optJ2',
    spin_polarized=False,
    identifier = 'opt',
    job = jobs['qmc'],
    pseudos = ['C.BFD.xml'],
    system = dia16,
    input_type = 'basic',
    twistnum   = 0,
    corrections = [],
    jastrows = [('J1','bspline',6),
                ('J2','bspline',6)],
    calculations = optims,
    dependencies = (conv,'orbitals')
)

dmc_run = generate_qmcpack(
    path = 'diamond/production',
    spin_polarized=False,
    identifier = 'qmc',
    job = jobs['qmc'],
    pseudos = ['C.BFD.xml'],
    system = dia16,
    input_type = 'basic',
    estimators = [],
    corrections = [],
    jastrows = [],
    calculations = [
        vmc(
            timestep         = 0.3,
            warmupsteps      = 10,
            blocks           = 80,
            steps            = 5,
            substeps         = 3,
            #samplesperthread = 10,
            samples          = 2048,
        ),
        dmc(
            timestep         = 0.01,
            warmupsteps      = 10,
            blocks           = 80,
            steps            = 5,
            nonlocalmoves    = True,
        ),
        dmc(
            timestep         = 0.005,
            warmupsteps      = 50,
            blocks           = 80,
            steps            = 5,
            nonlocalmoves    = True,
        ),
    ],
    dependencies = [(conv,'orbitals'),
                    (optJ2,'jastrow')]
)


run_project(scf,conv,qmc,optJ2,dmc_run)


