"""Octant is a suite of tools for working with C-grid ocean models.

"""

classifiers = """\
Development Status :: beta
Environment :: Console
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: MIT
Operating System :: OS Independent
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules
"""

from numpy.distutils.core import Extension

iso = Extension(name = '_iso',
                sources = ['octant/src/iso.f'])

doclines = __doc__.split("\n")

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(name = "octant",
          version = '0.1.0',
          description = doclines[0],
          long_description = "\n".join(doclines[2:]),
          author = "Robert Hetland",
          author_email = "hetland@tamu.edu",
          url = "http://octant.googlecode.com/",
          packages = ['octant',
                      'octant.extern',
                      'octant.ocean'],
          license = 'BSD',
          platforms = ["any"],
          ext_package='octant',
          ext_modules = [iso, ],
          classifiers = filter(None, classifiers.split("\n")),
          )
    
