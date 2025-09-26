# Author : Byunghun Hwang <bh.hwang@iae.re.kr>


# Build for architecture selection (editable!!)
ARCH := $(shell uname -m)
OS := $(shell uname)

CURRENT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CURRENT_DIR_NAME := $(notdir $(patsubst %/,%,$(dir $(CURRENT_DIR))))

# path
FLAME_PATH = $(CURRENT_DIR)/flame
DEP_PATH = $(CURRENT_DIR)/dep
INCLUDES = $(FLAME_PATH)/include
SOURCE_FILES = .

#Compilers
ifeq ($(ARCH),arm64)
	CC := g++
	GCC := gcc
	LD_LIBRARY_PATH += -L./lib/arm64
	OUTDIR		= $(CURRENT_DIR)/bin/arm64/
	BUILDDIR	= $(CURRENT_DIR)/bin/arm64/
	INCLUDE_DIR = -I./ -I$(CURRENT_DIR)/ -I$(CURRENT_DIR)/include/ -I$(CURRENT_DIR)/include/dep -I/usr/include
	LD_LIBRARY_PATH += -L/usr/local/lib -L./lib/arm64/
else ifeq ($(ARCH), armhf)
	CC := /usr/bin/arm-linux-gnueabihf-g++-9
	GCC := /usr/bin/arm-linux-gnueabihf-gcc-9
	LD_LIBRARY_PATH += -L./lib/armhf
	OUTDIR		= $(CURRENT_DIR)/bin/armhf/
	BUILDDIR	= $(CURRENT_DIR)/bin/armhf/
	INCLUDE_DIR = -I./ -I$(CURRENT_DIR)/ -I$(CURRENT_DIR)/include/ -I$(CURRENT_DIR)/include/dep -I/usr/include
	LD_LIBRARY_PATH += -L/usr/local/lib -L./lib/armhf/
else ifeq ($(ARCH), aarch64) # for Mac Apple Silicon
	CC := g++
	GCC := gcc
#	LD_LIBRARY_PATH += -L./lib/aarch64-linux-gnu
	OUTDIR		= $(CURRENT_DIR)/bin/aarch64/
	BUILDDIR	= $(CURRENT_DIR)/bin/aarch64/
	INCLUDE_DIR = -I./ -I$(CURRENT_DIR) -I$(FLAME_PATH)/include -I$(CURRENT_DIR)/include/dep -I/usr/include -I/usr/local/include -I/usr/include/opencv4
	LIBDIR = -L/usr/local/lib -L$(CURRENT_DIR)/lib/aarch64-linux-gnu/
export LD_LIBRARY_PATH := $(LIBDIR):$(LD_LIBRARY_PATH)
else
	CC := g++
	GCC := gcc
#	LD_LIBRARY_PATH += -L./lib/x86_64
	OUTDIR		= $(CURRENT_DIR)/bin/x86_64/
	BUILDDIR	= $(CURRENT_DIR)/bin/x86_64/
	INCLUDE_DIR = -I./ -I$(CURRENT_DIR) -I$(FLAME_PATH)/include -I$(FLAME_PATH)/include/dep -I/usr/include -I/usr/local/include -I/usr/include/opencv4
	LIBDIR = -L/usr/local/lib -L$(FLAME_PATH)/lib/x86_64/ -L/usr/lib/x86-64-linux-gnu
export LD_LIBRARY_PATH := $(LIBDIR):$(LD_LIBRARY_PATH)
endif

# OS
ifeq ($(OS),Linux) #for Linux
	LDFLAGS = -Wl,--export-dynamic -Wl,-rpath=. $(LIBDIR) -L$(LIBDIR)
	LDLIBS = -pthread -lrt -ldl -lm -lzmq
endif



$(shell mkdir -p $(OUTDIR))
$(shell mkdir -p $(BUILDDIR))
REV_COUNT = $(shell git rev-list --all --count)
MIN_COUNT = 0 #$(shell git tag | wc -l)

#if release(-O3), debug(-O0)
# if release mode compile, remove -DNDEBUG
CXXFLAGS = -O3 -fPIC -Wall -std=c++20 -D__cplusplus=202002L -Wno-deprecated-enum-enum-conversion

#custom definitions
CXXFLAGS += -D__MAJOR__=0 -D__MINOR__=$(MIN_COUNT) -D__REV__=$(REV_COUNT)
RM	= rm -rf


# flame service engine
flame:	$(BUILDDIR)flame.o \
		$(BUILDDIR)config.o \
		$(BUILDDIR)manager.o \
		$(BUILDDIR)driver.o \
		$(BUILDDIR)instance.o
		$(CC) $(LDFLAGS) $(LD_LIBRARY_PATH) -o $(BUILDDIR)$@ $^ $(LDLIBS)

$(BUILDDIR)flame.o:	$(FLAME_PATH)/tools/flame/flame.cc
					$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@
$(BUILDDIR)instance.o: $(FLAME_PATH)/tools/flame/instance.cc
						$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@
$(BUILDDIR)manager.o: $(FLAME_PATH)/tools/flame/manager.cc
						$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@
$(BUILDDIR)driver.o: $(INCLUDES)/flame/component/driver.cc
						$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@
$(BUILDDIR)config.o: $(INCLUDES)/flame/config.cc
						$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@


# components

uvc_camera_grabber.comp:	$(BUILDDIR)uvc.camera.grabber.o \
							$(BUILDDIR)support.o
							$(CC) $(LDFLAGS) $(LD_LIBRARY_PATH) -shared -o $(BUILDDIR)/osm/$@ $^ $(LDFLAGS) $(LDLIBS) -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio
$(BUILDDIR)uvc.camera.grabber.o:	$(CURRENT_DIR)/components/uvc.camera.grabber/uvc.camera.grabber.cc
									$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@
$(BUILDDIR)support.o:	$(CURRENT_DIR)/components/uvc.camera.grabber/support.cc
									$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@

solectrix_camera_grabber.comp:	$(BUILDDIR)solectrix.camera.grabber.o \
								$(BUILDDIR)sxpf_grabber.o \
								$(BUILDDIR)img_decode.o
							$(CC) $(LDFLAGS) $(LD_LIBRARY_PATH) -shared -o $(BUILDDIR)/osm/$@ $^ $(LDFLAGS) $(LDLIBS) -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lsxpf_ll
$(BUILDDIR)solectrix.camera.grabber.o:	$(CURRENT_DIR)/components/solectrix.camera.grabber/solectrix.camera.grabber.cc
									$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@
$(BUILDDIR)sxpf_grabber.o:	$(CURRENT_DIR)/components/solectrix.camera.grabber/sxpf_grabber.cc
									$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@
$(BUILDDIR)img_decode.o:	$(CURRENT_DIR)/components/solectrix.camera.grabber/img_decode.cc
									$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@

kvaser_can_interface.comp:	$(BUILDDIR)kvaser.can.interface.o
							$(CC) $(LDFLAGS) $(LD_LIBRARY_PATH) -shared -o $(BUILDDIR)/osm/$@ $^ $(LDFLAGS) $(LDLIBS) -lcanlib
$(BUILDDIR)kvaser.can.interface.o:	$(CURRENT_DIR)/components/kvaser.can.interface/kvaser.can.interface.cc
									$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@

body_kps_inference.comp:	$(BUILDDIR)body.kps.inference.o
							$(CC) $(LDFLAGS) $(LD_LIBRARY_PATH) -shared -o $(BUILDDIR)/osm/$@ $^ $(LDFLAGS) $(LDLIBS) -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lnvinfer -lnvonnxparser -lcudart -lcublas
$(BUILDDIR)body.kps.inference.o:	$(CURRENT_DIR)/components/body.kps.inference/body.kps.inference.cc
									$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@


os_model_inference.comp:	$(BUILDDIR)os.model.inference.o
							$(CC) $(LDFLAGS) $(LD_LIBRARY_PATH) -shared -o $(BUILDDIR)/osm/$@ $^ $(LDFLAGS) $(LDLIBS)
$(BUILDDIR)os.model.inference.o:	$(CURRENT_DIR)/components/os.model.inference/os.model.inference.cc
									$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@

headpose_model_inference.comp:	$(BUILDDIR)headpose.model.inference.o
							$(CC) $(LDFLAGS) $(LD_LIBRARY_PATH) -shared -o $(BUILDDIR)/osm/$@ $^ $(LDFLAGS) $(LDLIBS) -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_calib3d -lmediapipe
$(BUILDDIR)headpose.model.inference.o:	$(CURRENT_DIR)/components/headpose.model.inference/headpose.model.inference.cc
									$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@



video_file_grabber.comp:	$(BUILDDIR)video.file.grabber.o
							$(CC) $(LDFLAGS) $(LD_LIBRARY_PATH) -shared -o $(BUILDDIR)/osm/$@ $^ $(LDFLAGS) $(LDLIBS) -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio
$(BUILDDIR)video.file.grabber.o:	$(CURRENT_DIR)/components/video.file.grabber/video.file.grabber.cc
									$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@


all : flame

osm : flame uvc_camera_grabber.comp solectrix_camera_grabber.comp body_kps_inference.comp os_model_inference.comp headpose_model_inference.comp

deploy : FORCE
	cp $(BUILDDIR)/*.comp $(BUILDDIR)/flame $(BINDIR)
clean : FORCE 
		$(RM) $(BUILDDIR)/*.o $(BUILDDIR)/*.comp $(BUILDDIR)/osm/*.comp $(BUILDDIR)/flame
debug:
	@echo "Building for Architecture : $(ARCH)"
	@echo "Building for OS : $(OS)"

FORCE : 