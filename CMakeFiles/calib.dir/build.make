# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/junior/Desktop/cube

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/junior/Desktop/cube

# Include any dependencies generated for this target.
include CMakeFiles/calib.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/calib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/calib.dir/flags.make

CMakeFiles/calib.dir/match.cpp.o: CMakeFiles/calib.dir/flags.make
CMakeFiles/calib.dir/match.cpp.o: match.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/junior/Desktop/cube/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/calib.dir/match.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/calib.dir/match.cpp.o -c /home/junior/Desktop/cube/match.cpp

CMakeFiles/calib.dir/match.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/calib.dir/match.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/junior/Desktop/cube/match.cpp > CMakeFiles/calib.dir/match.cpp.i

CMakeFiles/calib.dir/match.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/calib.dir/match.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/junior/Desktop/cube/match.cpp -o CMakeFiles/calib.dir/match.cpp.s

CMakeFiles/calib.dir/match.cpp.o.requires:
.PHONY : CMakeFiles/calib.dir/match.cpp.o.requires

CMakeFiles/calib.dir/match.cpp.o.provides: CMakeFiles/calib.dir/match.cpp.o.requires
	$(MAKE) -f CMakeFiles/calib.dir/build.make CMakeFiles/calib.dir/match.cpp.o.provides.build
.PHONY : CMakeFiles/calib.dir/match.cpp.o.provides

CMakeFiles/calib.dir/match.cpp.o.provides.build: CMakeFiles/calib.dir/match.cpp.o

# Object files for target calib
calib_OBJECTS = \
"CMakeFiles/calib.dir/match.cpp.o"

# External object files for target calib
calib_EXTERNAL_OBJECTS =

calib: CMakeFiles/calib.dir/match.cpp.o
calib: CMakeFiles/calib.dir/build.make
calib: /usr/local/lib/libopencv_videostab.so.2.4.9
calib: /usr/local/lib/libopencv_video.so.2.4.9
calib: /usr/local/lib/libopencv_ts.a
calib: /usr/local/lib/libopencv_superres.so.2.4.9
calib: /usr/local/lib/libopencv_stitching.so.2.4.9
calib: /usr/local/lib/libopencv_photo.so.2.4.9
calib: /usr/local/lib/libopencv_ocl.so.2.4.9
calib: /usr/local/lib/libopencv_objdetect.so.2.4.9
calib: /usr/local/lib/libopencv_nonfree.so.2.4.9
calib: /usr/local/lib/libopencv_ml.so.2.4.9
calib: /usr/local/lib/libopencv_legacy.so.2.4.9
calib: /usr/local/lib/libopencv_imgproc.so.2.4.9
calib: /usr/local/lib/libopencv_highgui.so.2.4.9
calib: /usr/local/lib/libopencv_gpu.so.2.4.9
calib: /usr/local/lib/libopencv_flann.so.2.4.9
calib: /usr/local/lib/libopencv_features2d.so.2.4.9
calib: /usr/local/lib/libopencv_core.so.2.4.9
calib: /usr/local/lib/libopencv_contrib.so.2.4.9
calib: /usr/local/lib/libopencv_calib3d.so.2.4.9
calib: /usr/lib/x86_64-linux-gnu/libGLU.so
calib: /usr/lib/x86_64-linux-gnu/libGL.so
calib: /usr/lib/x86_64-linux-gnu/libSM.so
calib: /usr/lib/x86_64-linux-gnu/libICE.so
calib: /usr/lib/x86_64-linux-gnu/libX11.so
calib: /usr/lib/x86_64-linux-gnu/libXext.so
calib: /usr/lib/x86_64-linux-gnu/libglut.so
calib: /usr/lib/x86_64-linux-gnu/libXmu.so
calib: /usr/lib/x86_64-linux-gnu/libXi.so
calib: /usr/lib/x86_64-linux-gnu/libGLU.so
calib: /usr/lib/x86_64-linux-gnu/libGL.so
calib: /usr/lib/x86_64-linux-gnu/libSM.so
calib: /usr/lib/x86_64-linux-gnu/libICE.so
calib: /usr/lib/x86_64-linux-gnu/libX11.so
calib: /usr/lib/x86_64-linux-gnu/libXext.so
calib: /usr/local/lib/libopencv_nonfree.so.2.4.9
calib: /usr/local/lib/libopencv_ocl.so.2.4.9
calib: /usr/local/lib/libopencv_gpu.so.2.4.9
calib: /usr/local/lib/libopencv_photo.so.2.4.9
calib: /usr/local/lib/libopencv_objdetect.so.2.4.9
calib: /usr/local/lib/libopencv_legacy.so.2.4.9
calib: /usr/local/lib/libopencv_video.so.2.4.9
calib: /usr/local/lib/libopencv_ml.so.2.4.9
calib: /usr/local/lib/libopencv_calib3d.so.2.4.9
calib: /usr/local/lib/libopencv_features2d.so.2.4.9
calib: /usr/local/lib/libopencv_highgui.so.2.4.9
calib: /usr/local/lib/libopencv_imgproc.so.2.4.9
calib: /usr/local/lib/libopencv_flann.so.2.4.9
calib: /usr/local/lib/libopencv_core.so.2.4.9
calib: CMakeFiles/calib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable calib"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/calib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/calib.dir/build: calib
.PHONY : CMakeFiles/calib.dir/build

CMakeFiles/calib.dir/requires: CMakeFiles/calib.dir/match.cpp.o.requires
.PHONY : CMakeFiles/calib.dir/requires

CMakeFiles/calib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/calib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/calib.dir/clean

CMakeFiles/calib.dir/depend:
	cd /home/junior/Desktop/cube && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/junior/Desktop/cube /home/junior/Desktop/cube /home/junior/Desktop/cube /home/junior/Desktop/cube /home/junior/Desktop/cube/CMakeFiles/calib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/calib.dir/depend

