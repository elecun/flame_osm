# Installing Kvaser CANlib SDK

## System requirements
* The package libxml2-dev is needed in order to build kvamemolibxml.
* The package zlib1g-dev is needed to build kvlclib.
```
$ sudo apt-get install libxml2-dev zlib1g-dev
```

## Install
* To download and unpack the latest version of Linux SDK library use:
```
$ wget --content-disposition "https://www.kvaser.com/downloads-kvaser/?utm_source=software&utm_ean=7330130981966&utm_status=latest"
$ tar xvzf kvlibsdk.tar.gz
$ cd kvlibsdk
```
* To build everything, run
```
$ make
```
* To run self-tests, run
```
$ make check
```
* To install everything, run
```
$ sudo make install
```

## Uninstall
* To uninstall everything, run
```
$ sudo make uninstall 
```

## reference
1. https://kvaser.com/canlib-webhelp/section_install_linux.htm