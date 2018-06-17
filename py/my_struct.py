__author__ = 'zsh'

import os
import struct

fp = open('test.bin', 'wb')

name = b'zsh'
age = 33
sex = b'man'
job = b'programer'

fp.write(struct.pack('3si3s9s', name, age, sex, job))
fp.flush()
fp.close()

fd = open('test.bin', 'rb')
print(struct.unpack('3si3s9s', fd.read(20)))
fd.close()