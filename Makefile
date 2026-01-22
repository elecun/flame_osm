# Author : Byunghun Hwang <bh.hwang@iae.re.kr>

# -----------------------------------------------------------------------------
# Architecture & OS Detection
# -----------------------------------------------------------------------------
ARCH := $(shell uname -m)
OS := $(shell uname)

# -----------------------------------------------------------------------------
# Paths & Directories
# -----------------------------------------------------------------------------
CURRENT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CURRENT_DIR_NAME := $(notdir $(patsubst %/,%,$(dir $(CURRENT_DIR))))

FLAME_PATH = $(CURRENT_DIR)/flame
DEP_PATH = $(CURRENT_DIR)/dep
INCLUDES_PATH = $(FLAME_PATH)/include

# Output Directories
# Default output directory is based on architecture
OUTDIR		= $(CURRENT_DIR)/bin/$(ARCH)/
BUILDDIR	= $(CURRENT_DIR)/bin/$(ARCH)/

# -----------------------------------------------------------------------------
# Compiler & Flags Settings
# -----------------------------------------------------------------------------
# Defaults
CC := g++
GCC := gcc
RM	= rm -rf

# Base Flags
# if release mode compile (-O3), remove -DNDEBUG if previously set (implied)
CXXFLAGS = -O3 -fPIC -Wall -std=c++20 -D__cplusplus=202002L -Wno-deprecated-enum-enum-conversion

# Git Revisions
REV_COUNT = 0 #$(shell git rev-list --all --count)
MIN_COUNT = 0 #$(shell git tag | wc -l)
CXXFLAGS += -D__MAJOR__=0 -D__MINOR__=$(MIN_COUNT) -D__REV__=$(REV_COUNT)

# Common Includes and Libraries setup
INCLUDE_BASE = -I./ -I$(CURRENT_DIR) -I/usr/include
LDFLAGS_COMMON = 

# Platform Specific Configuration
ifeq ($(ARCH),arm64)
	# ARM64
	INCLUDE_DIR = $(INCLUDE_BASE) -I$(CURRENT_DIR)/include/ -I$(CURRENT_DIR)/include/dep
	LIBDIR = -L/usr/local/lib -L./lib/arm64

else ifeq ($(ARCH), armhf)
	# ARMHF
	CC := /usr/bin/arm-linux-gnueabihf-g++-9
	GCC := /usr/bin/arm-linux-gnueabihf-gcc-9
	INCLUDE_DIR = $(INCLUDE_BASE) -I$(CURRENT_DIR)/include/ -I$(CURRENT_DIR)/include/dep
	LIBDIR = -L/usr/local/lib -L./lib/armhf

else ifeq ($(ARCH), aarch64) 
	# AARCH64
	INCLUDE_DIR = $(INCLUDE_BASE) -I$(FLAME_PATH)/include -I$(CURRENT_DIR)/include/dep -I/usr/local/include -I/usr/include/opencv4
	LIBDIR = -L/usr/local/lib -L$(CURRENT_DIR)/lib/aarch64-linux-gnu/

else
	# x86_64 (Default)
	INCLUDE_DIR = $(INCLUDE_BASE) -I$(FLAME_PATH)/include -I$(FLAME_PATH)/include/dep -I/usr/local/include -I/usr/include/opencv4 -I/usr/local/cuda/include
	LIBDIR = -L/usr/local/lib -L$(FLAME_PATH)/lib/x86_64/ -L/usr/lib/x86-64-linux-gnu -L/usr/local/cuda/lib64
endif

# LDFLAGS definition
# Always include LIBDIR
LDFLAGS = $(LIBDIR)

# OS Specific Flags
ifeq ($(OS),Linux)
	LDFLAGS += -Wl,--export-dynamic -Wl,-rpath=. 
	LDLIBS = -pthread -lrt -ldl -lm -lzmq
endif

# -----------------------------------------------------------------------------
# Rules
# -----------------------------------------------------------------------------

# Ensure directories exist (Pre-requisite for targets)
$(shell mkdir -p $(OUTDIR))
$(shell mkdir -p $(BUILDDIR))

.PHONY: all clean debug deploy FORCE osm

all : flame

# -----------------------------------------------------------------------------
# Target: flame (Service Engine)
# -----------------------------------------------------------------------------
FLAME_OBJS = \
	$(BUILDDIR)flame.o \
	$(BUILDDIR)config.o \
	$(BUILDDIR)manager.o \
	$(BUILDDIR)driver.o \
	$(BUILDDIR)instance.o

flame: $(FLAME_OBJS)
	$(CC) $(LDFLAGS) -o $(BUILDDIR)$@ $^ $(LDLIBS)

$(BUILDDIR)flame.o: $(FLAME_PATH)/tools/flame/flame.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

$(BUILDDIR)instance.o: $(FLAME_PATH)/tools/flame/instance.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

$(BUILDDIR)manager.o: $(FLAME_PATH)/tools/flame/manager.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

$(BUILDDIR)driver.o: $(INCLUDES_PATH)/flame/component/driver.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

$(BUILDDIR)config.o: $(INCLUDES_PATH)/flame/config.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@


# -----------------------------------------------------------------------------
# Targets: Components
# -----------------------------------------------------------------------------

# UVC Camera Grabber
uvc_camera_grabber.comp: $(BUILDDIR)uvc.camera.grabber.o $(BUILDDIR)support.o
	$(CC) $(LDFLAGS) -shared -o $(BUILDDIR)/osm/$@ $^ $(LDFLAGS) $(LDLIBS) -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio

$(BUILDDIR)uvc.camera.grabber.o: $(CURRENT_DIR)/components/uvc.camera.grabber/uvc.camera.grabber.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

$(BUILDDIR)support.o: $(CURRENT_DIR)/components/uvc.camera.grabber/support.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

# Solectrix Camera Grabber
SOLECTRIX_OBJS = \
	$(BUILDDIR)solectrix.camera.grabber.o \
	$(BUILDDIR)sxpf_grabber.o \
	$(BUILDDIR)img_decode.o \
	$(BUILDDIR)core_frame_processing.o

solectrix_camera_grabber.comp: $(SOLECTRIX_OBJS)
	$(CC) $(LDFLAGS) -shared -o $(BUILDDIR)/osm/$@ $^ $(LDFLAGS) $(LDLIBS) -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lsxpf_ll

$(BUILDDIR)solectrix.camera.grabber.o: $(CURRENT_DIR)/components/solectrix.camera.grabber/solectrix.camera.grabber.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

$(BUILDDIR)sxpf_grabber.o: $(CURRENT_DIR)/components/solectrix.camera.grabber/sxpf_grabber.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

$(BUILDDIR)img_decode.o: $(CURRENT_DIR)/components/solectrix.camera.grabber/img_decode.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

$(BUILDDIR)core_frame_processing.o: $(CURRENT_DIR)/components/solectrix.camera.grabber/core_frame_processing.cpp
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

# Kvaser CAN Interface
kvaser_can_interface.comp: $(BUILDDIR)kvaser.can.interface.o
	$(CC) $(LDFLAGS) -shared -o $(BUILDDIR)/osm/$@ $^ $(LDFLAGS) $(LDLIBS) -lcanlib

$(BUILDDIR)kvaser.can.interface.o: $(CURRENT_DIR)/components/kvaser.can.interface/kvaser.can.interface.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

# Body KPS Inference
body_kps_inference.comp: $(BUILDDIR)body.kps.inference.o
	$(CC) $(LDFLAGS) -shared -o $(BUILDDIR)/osm/$@ $^ $(LDFLAGS) $(LDLIBS) -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lnvinfer -lnvonnxparser -lcudart -lcublas

$(BUILDDIR)body.kps.inference.o: $(CURRENT_DIR)/components/body.kps.inference/body.kps.inference.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

# Face Detection Inference
face_detection_inference.comp: $(BUILDDIR)face.detection.inference.o
	$(CC) $(LDFLAGS) -shared -o $(BUILDDIR)/osm/$@ $^ $(LDFLAGS) $(LDLIBS) -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lnvinfer -lnvonnxparser -lcudart -lcublas

$(BUILDDIR)face.detection.inference.o: $(CURRENT_DIR)/components/face.detection.inference/face.detection.inference.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

# OS Model Inference
os_model_inference.comp: $(BUILDDIR)os.model.inference.o
	$(CC) $(LDFLAGS) -shared -o $(BUILDDIR)/osm/$@ $^ $(LDFLAGS) $(LDLIBS)

$(BUILDDIR)os.model.inference.o: $(CURRENT_DIR)/components/os.model.inference/os.model.inference.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

# Headpose Model Inference
headpose_model_inference.comp: $(BUILDDIR)headpose.model.inference.o
	$(CC) $(LDFLAGS) -shared -o $(BUILDDIR)/osm/$@ $^ $(LDFLAGS) $(LDLIBS) -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_calib3d -lmediapipe

$(BUILDDIR)headpose.model.inference.o: $(CURRENT_DIR)/components/headpose.model.inference/headpose.model.inference.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

# Video File Grabber
video_file_grabber.comp: $(BUILDDIR)video.file.grabber.o
	$(CC) $(LDFLAGS) -shared -o $(BUILDDIR)/osm/$@ $^ $(LDFLAGS) $(LDLIBS) -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio

$(BUILDDIR)video.file.grabber.o: $(CURRENT_DIR)/components/video.file.grabber/video.file.grabber.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

# OSM Group Target
osm : flame uvc_camera_grabber.comp solectrix_camera_grabber.comp body_kps_inference.comp os_model_inference.comp headpose_model_inference.comp video_file_grabber.comp face_detection_inference.comp

deploy : FORCE
	cp $(BUILDDIR)/*.comp $(BUILDDIR)/flame $(BINDIR)

clean : FORCE 
	$(RM) $(BUILDDIR)/*.o $(BUILDDIR)/*.comp $(BUILDDIR)/osm/*.comp $(BUILDDIR)/flame

debug:
	@echo "Building for Architecture : $(ARCH)"
	@echo "Building for OS : $(OS)"
	@echo "LDFLAGS: $(LDFLAGS)"

FORCE :