"""Single authoritative environment lock; setup uses these exact requirements."""

ENVIRONMENT_NAME="'QC'"
PYTHON_VERSION='3.13.13'
PACKAGE_VERSIONS={
    'qiskit':'2.5.2',
    'qiskit-aer':'0.17.2',
    'qiskit-algorithms':'0.4.0',
    'numpy':'2.4.6',
    'scipy':'1.17.1',
    'psutil':'7.2.2',
}


def requirements():
    return [name+'=='+version for name,version in PACKAGE_VERSIONS.items()]


def requirements_text():
    return '# Generated from qc_versions.py; edit that Python authority, then export.\n'+'\n'.join(requirements())+'\n'


if __name__=='__main__':
    # Display only; setup consumes requirements() directly, without text files.
    print(requirements_text(),end='')
