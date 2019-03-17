#! /usr/bin/env python
import os
from nexus import settings,job,run_project,obj
from nexus import generate_physical_system
from nexus import generate_pwscf
from machine_configs import get_taito_configs

settings(
    pseudo_dir    = './pseudopotentials',
    results       = '',
    status_only   = 0,
    generate_only = 1, 
    sleep         = 3,
    machine       = 'taito',
)

jobs = get_taito_configs()

cubic_box_size=[10.0]
x=1.0*cubic_box_size[0]
d=1.2074 # nuclei separation in Angstrom

O2 = generate_physical_system(
    units  = 'A',
    axes   = [[ x,   0.0 ,  0.0   ],
              [ 0.,   x  ,  0.0   ],
              [ 0.,   0. ,   x    ]],
    elem   = ['O','O'],
    pos    = [[ x/2-d/2    ,  x/2    ,  x/2    ],
              [ x/2+d/2    ,  x/2    ,  x/2    ]],
    net_spin  = 0,
    tiling    = (1,1,1),
    kgrid     = (1,1,1), # scf kgrid given below to enable symmetries
    kshift    = (0,0,0),
    O         = 6,
)

scf = generate_pwscf(
    identifier   = 'scf',
    path         = 'scf',
    job          = jobs['scf'],
    input_type   = 'generic',
    calculation  = 'scf',
    input_dft    = 'lda', 
    ecutwfc      = 200,   
    conv_thr     = 1e-8, 
    nosym        = True,
    wf_collect   = True,
    system       = O2,
    kgrid        = (1,1,1),
    pseudos      = ['O.BFD.upf'], 
    )

run_project(scf)

