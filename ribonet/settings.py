import os

try:
    BASE_DIR = os.environ['RIBONETHOME']
except:
    raise Exception('please set RIBONETHOME environment variable')

try:
    WORK_DIR = '%s/nn' % os.environ['WORK']
except:
    WORK_DIR = BASE_DIR

def set_molecule(mol):
    global paramfile
    paramfile = params[mol]
    global bases
    global baselist
    global bps
    global mismatches
    if mol == 'rna':
        bases = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        baselist = ['A', 'C', 'G', 'U']  # preserve order
        bps = ['AU', 'CG', 'GC', 'UA', 'GU', 'UG']
        mismatches = ['AA', 'AC', 'AG', 'AU', 'CA', 'CC', 'CG', 'CU',
                      'GA', 'GC', 'GG', 'GU', 'UA', 'UC', 'UG', 'UU']
    else:
        bases = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        baselist = ['A', 'C', 'G', 'T']  # preserve order
        bps = ['AT', 'CG', 'GC', 'TA', 'GT', 'TG']
        mismatches = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
                      'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']

# numbering scheme for bases
params = {'dna': 'dna1998', 'rna': 'rna1995'}
molecule = 'dna'
set_molecule(molecule)

# directories
LOGS_DIR = '%s/logs' % WORK_DIR
RESULTS_DIR = '%s/results' % WORK_DIR
MODELS_DIR = '%s/models' % WORK_DIR
RESOURCES_DIR = '%s/resources' % BASE_DIR
PARAMS_DIR = '%s/nupack/parameters' % RESOURCES_DIR
TEMP_DIR = '%s/tmp' % WORK_DIR

XDIM = 8
