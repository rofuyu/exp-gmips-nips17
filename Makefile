CXX?=g++
VERSION=0.1

all: 
	make -C src

zip:
	make clean; cd ../;  zip -r --symlinks gmips-nips17-exp-${VERSION}.zip release-${VERSION}/

clean:
	make -C src clean; rm -rf *pyc .*~ data/.*~

